import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, List, Tuple, Dict
import traceback
import math
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
)
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation

import PIL.Image
import PIL.ImageOps
import copy
from loguru import logger
import traceback

from open_r1.trainer.grpo_trainer_debug import Qwen2VLGRPOTrainer
from open_r1.utils import extract_bbox, modify_string_with_new_data, extract_action
from open_r1.reward import length_reward_dast
import cropping_utils



class Qwen2VLGRPOTrainerWithCropping(Qwen2VLGRPOTrainer):
    def __init__(self,
                 *args,
                 crop_factor: float = 0.6,
                 max_crop_attempts: int = 4,
                 reward_threshold: float = 0.0,
                 grounding_reward_key: str = "rewards/grounding_reward",
                 action_reward_key: str = "rewards/action_reward",
                 thinking_strategy: str = "dast",
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.crop_factor = crop_factor
        self.max_crop_attempts = max_crop_attempts
        self.reward_threshold = reward_threshold
        self.grounding_reward_key = grounding_reward_key
        self.action_reward_key = action_reward_key
        self.thinking_strategy = thinking_strategy
        self.attempt_num = defaultdict(list)  # Track attempt number for each prompt, classified by epoch
        self.failed_items = []
        # logger.info(f"Cropping enabled: factor={self.crop_factor}, max_attempts={self.max_crop_attempts}, threshold={self.reward_threshold}, reward_key='{self.grounding_reward_key}'")
        self.main_info(f"Cropping enabled: factor={self.crop_factor}, max_attempts={self.max_crop_attempts}, threshold={self.reward_threshold}, reward_key='{self.grounding_reward_key}'")

    def main_debug(self, msg):
        if self.accelerator.is_main_process:
            logger.debug(msg)
    def main_info(self, msg):
        if self.accelerator.is_main_process:
            logger.info(msg)
    def main_warning(self, msg):
        if self.accelerator.is_main_process:
            logger.warning(msg)
    def main_error(self, msg):
        if self.accelerator.is_main_process:
            logger.error(msg)
            logger.error(f"Error traceback: {traceback.format_exc()}")



    def _extract_bboxes_from_inputs(self, inputs: List[Dict]) -> List[Optional[List[int]]]:
        bboxes = []
        for item in inputs:
            bbox = None
            if "solution" in item and extract_action(item["solution"]) == "click":
                if  isinstance(item["solution"], str):
                    extracted_bbox_info, _ = extract_bbox(item["solution"])
                    if extracted_bbox_info and isinstance(extracted_bbox_info, list) and len(extracted_bbox_info) == 4:
                        bbox = [int(c) for c in extracted_bbox_info] # Ensure int
                elif "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4:
                    bbox = [int(c) for c in item["bbox"]]
            bboxes.append(bbox)
        return bboxes

    def _adapt_parent_generate_and_score(
        self, inputs_for_attempt: List[Dict], model_to_generate_with: PreTrainedModel
    ) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
        """
        This method carefully replicates the logic of the parent's 
        _generate_and_score_completions method but returns metrics 
        instead of appending to self._metrics directly.
        """

        device = self.accelerator.device
        prompts = [x["problem"] for x in inputs_for_attempt] # Original prompt structures
        prompts_text = [ # Text version of prompts for tokenization
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs_for_attempt
        ]

        images = []
        original_image_sizes = []
        for x in inputs_for_attempt: # inputs_for_attempt now contains potentially cropped images
            img_pil = x.get("image") # Should be a PIL image from _prepare_images_for_attempt
            if img_pil and isinstance(img_pil, PIL.Image.Image):
                original_image_sizes.append(img_pil.size)
                images.append(img_pil)
            else: # No image or not a PIL image
                raise ValueError(f"Expected a PIL image in input, but got: {img_pil} for prompt: {x['prompt']}")
        
        batch_has_images = any(img is not None for img in images)
        if batch_has_images:
            prompt_inputs = self.processing_class(text=prompts_text, images=images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        else:
            prompt_inputs = self.processing_class(text=prompts_text, images=None, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        
        # Call to _prepare_inputs from Trainer class (moves to device)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        for key, value in prompt_inputs.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                prompt_inputs[key] = value.to(device) 

        scales_for_rewards = []
        if batch_has_images and 'image_grid_thw' in prompt_inputs and prompt_inputs['image_grid_thw'] is not None:
            patch_size = self.processing_class.image_processor.patch_size
            for i in range(len(inputs_for_attempt)):
                if images[i] is not None and i < len(prompt_inputs['image_grid_thw']):
                    orig_w, orig_h = original_image_sizes[i]
                    if orig_w == 0 or orig_h == 0: raise ValueError(f"Original image size for item {i} is zero: {original_image_sizes[i]}")
                    current_grid_thw = prompt_inputs['image_grid_thw'][i]
                    resized_h = current_grid_thw[1].item() * patch_size
                    resized_w = current_grid_thw[2].item() * patch_size
                    scales_for_rewards.append([orig_w / resized_w, orig_h / resized_h])
                else: raise ValueError(f"index {i} out of bounds for image_grid_thw in prompt_inputs. Ensure all images are processed correctly.")
        else: raise ValueError("Expected image_grid_thw in prompt_inputs for cropping rewards")

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        pixel_values_gen = prompt_inputs.get("pixel_values")
        image_grid_thw_gen = prompt_inputs.get("image_grid_thw")
        # logger.debug(f"pixel_values_gen shape: {pixel_values_gen.shape if pixel_values_gen is not None else 'None'}")
        # logger.debug(f"image_grid_thw_gen shape: {image_grid_thw_gen.shape if image_grid_thw_gen is not None else 'None'}")


        # from transformers import AutoProcessor
        # qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct") 

        # prompts_text_qwen = [ # Text version of prompts for tokenization
        #     maybe_apply_chat_template(example, qwen_processor)["prompt"]
        #     for example in inputs_for_attempt
        # ]
        # qwen_inputs = qwen_processor(text=prompts_text_qwen, images=images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        # logger.debug(f"Qwen input_ids shape: {qwen_inputs['input_ids'].shape}") 
        # logger.debug(f"TARS input_ids shape: {prompt_inputs['input_ids'].shape}") 
        # logger.debug(f"Qwen attention_mask shape: {qwen_inputs['attention_mask'].shape}") 
        # logger.debug(f"TARS attention_mask shape: {prompt_inputs['attention_mask'].shape}")
        # logger.debug(f"generation_config: {self.generation_config}")

        # sampling for GRPO
        with unwrap_model_for_generation(model_to_generate_with, self.accelerator) as unwrapped_model:
            gen_kwargs = {"input_ids": prompt_ids, "attention_mask": prompt_mask, "generation_config": self.generation_config}
            if pixel_values_gen is not None: gen_kwargs["pixel_values"] = pixel_values_gen
            if image_grid_thw_gen is not None: gen_kwargs["image_grid_thw"] = image_grid_thw_gen 
            prompt_completion_ids = unwrapped_model.generate(**gen_kwargs)

        prompt_length = prompt_ids.size(1)
        actual_prompt_ids_from_generation = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        # logger.debug(f"completion_ids shape: {completion_ids.shape}\n completion_ids: {completion_ids}")

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        attention_mask_logp = torch.cat([prompt_mask, completion_mask], dim=1)
        input_ids_logp = prompt_completion_ids
        pixel_values_logp = prompt_inputs.get("pixel_values")
        image_grid_thw_logp = prompt_inputs.get("image_grid_thw")


        # loggprobs for GRPO
        with torch.no_grad():
            if self.num_iterations > 1: # self.num_iterations from GRPOConfig via parent
                old_per_token_logps = self._get_per_token_logps(model_to_generate_with, input_ids_logp, attention_mask_logp, pixel_values_logp, image_grid_thw_logp)
                old_per_token_logps = old_per_token_logps[:, prompt_length -1:]
            else: old_per_token_logps = None

            if self.beta == 0.0: ref_per_token_logps = None # self.beta from GRPOConfig
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids_logp, attention_mask_logp, pixel_values_logp, image_grid_thw_logp)
            else: # PEFT
                with self.accelerator.unwrap_model(model_to_generate_with).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model_to_generate_with, input_ids_logp, attention_mask_logp, pixel_values_logp, image_grid_thw_logp)
            if ref_per_token_logps is not None: ref_per_token_logps = ref_per_token_logps[:, prompt_length -1:]

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        logger.debug(f"Completions text: {completions_text}")
        completions_structured = [[{"role": "assistant", "content": c}] for c in completions_text] if is_conversational(inputs_for_attempt[0]) else completions_text

        # reward computation
        rewards_per_func_device = torch.zeros(len(prompts_text), len(self.reward_funcs), device=device)
        for i, (reward_fn_model, reward_proc_cls) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_fn_model, PreTrainedModel):
                reward_model_texts = []
                if is_conversational(inputs_for_attempt[0]):
                    for item, comp_part in zip(inputs_for_attempt, completions_structured):
                        full_conv = item["problem"] + comp_part
                        reward_model_texts.append(apply_chat_template({"messages": full_conv}, reward_proc_cls, add_generation_prompt=False)["text"])
                else:
                    reward_model_texts = [item["problem"] + comp for item, comp in zip(inputs_for_attempt, completions_structured)]
                
                reward_model_inputs = reward_proc_cls(reward_model_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
                reward_model_inputs = super()._prepare_inputs(reward_model_inputs) # Trainer's _prepare_inputs
                with torch.inference_mode():
                    rewards_per_func_device[:, i] = reward_fn_model(**reward_model_inputs).logits.squeeze(-1)
            else: # Custom reward function
                custom_prompts_arg = [item["problem"] for item in inputs_for_attempt]
                reward_kwargs = {key: [item.get(key) for item in inputs_for_attempt] for key in inputs_for_attempt[0].keys() if key not in ["problem", "completion"]}
                reward_kwargs["scales"] = scales_for_rewards
                output_reward_func = reward_fn_model(prompts=custom_prompts_arg, completions=completions_structured, **reward_kwargs)
                rewards_per_func_device[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                if reward_fn_model.__name__ == self.grounding_reward_key.split("/")[-1]:
                    logger.debug(f"Grounding reward function output: {output_reward_func} on device {device}")
        # logger.debug(f"shape of rewards_per_func_device: {rewards_per_func_device.shape}")
        all_rewards_per_func = self.accelerator.gather(rewards_per_func_device)
        """
        [
            func1:[R_A1, R_A2, R_A3, R_A4, R_B1, R_B2, R_B3, R_B4, R_C1, R_C2, R_C3, R_C4, R_D1, R_D2, R_D3, R_D4],
            func2:......
        ]
        """
        all_rewards_summed = all_rewards_per_func.sum(dim=1)
        grouped_rewards_per_func = all_rewards_per_func.view(-1, self.num_generations, len(self.reward_funcs)) # (unique_prompt_num, num_generation, num_reward_func) 
        grouped_mean_rewards_per_func = grouped_rewards_per_func.mean(dim=1).T # (num_reward_func, unique_prompt_num)
        if self.thinking_strategy=="dast":
            completion_token_length = torch.tensor(
                [completion_ids[i].size(0) for i in range(len(completion_ids))], device=device
            )
            completion_token_length = self.accelerator.gather(completion_token_length)
            len_rewards = length_reward_dast(all_rewards_summed, completion_token_length, self.num_generations)
            all_rewards_summed = all_rewards_summed + len_rewards
        num_total_items_global = all_rewards_summed.size(0)

        num_unique_prompts_global = num_total_items_global // self.num_generations
        if num_total_items_global % self.num_generations != 0: raise ValueError("Data alignment error with num_generations")
        grouped_rewards = all_rewards_summed.view(num_unique_prompts_global, self.num_generations)

        mean_grouped_rewards = grouped_rewards.mean(dim=1, keepdim=True)
        std_grouped_rewards = grouped_rewards.std(dim=1, keepdim=True)
        advantages_grouped = (grouped_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-5)
        advantages_all = advantages_grouped.reshape(-1)
        
        items_per_process = len(prompts_text)
        start_idx = self.accelerator.process_index * items_per_process
        end_idx = start_idx + items_per_process
        advantages_device = advantages_all[start_idx:end_idx]

        # Populate current_attempt_metrics instead of self._metrics
        current_attempt_metrics = defaultdict(list)
        grouped_mean_rewards_func_maps = [defaultdict(list) for _ in range(len(grouped_mean_rewards_per_func[1]))]
        current_attempt_metrics["completion_length"].append(self.accelerator.gather(completion_mask.sum(1)).float().mean().item())
        global_mean_rewards_per_func = all_rewards_per_func.mean(dim=0)
        for i, reward_fn_model in enumerate(self.reward_funcs):
            name = reward_fn_model.config._name_or_path.split("/")[-1] if isinstance(reward_fn_model, PreTrainedModel) else reward_fn_model.__name__
            # logger.debug(f"shape of global_mean_rewards_per_func[{i}]: {global_mean_rewards_per_func[i].shape}")
            current_attempt_metrics[f"rewards/{name}"].append(global_mean_rewards_per_func[i].item())
            for j in range(len(grouped_mean_rewards_per_func[1])):
                grouped_mean_rewards_func_maps[j][f"rewards/{name}"] = grouped_mean_rewards_per_func[i, j].item()


        if self.thinking_strategy=="dast":
            current_attempt_metrics[f"rewards/length_reward_dast"].append(len_rewards.mean(dim=0).item())
            # logger.debug(f"Reward for {name}: {global_mean_rewards_per_func[i].item():.4f}")
        current_attempt_metrics["reward"].append(all_rewards_summed.mean().item())
        current_attempt_metrics["reward_std"].append(std_grouped_rewards.mean().item())
        # --- End of adapted logic ---

        results_for_loss = {
            "prompt_ids": actual_prompt_ids_from_generation,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages_device,
            "pixel_values": pixel_values_logp,
            "image_grid_thw": image_grid_thw_logp,
        }
        # logger.debug(f"Results for loss: {results_for_loss.keys()}")
        return results_for_loss, current_attempt_metrics, grouped_mean_rewards_func_maps


    def _prepare_images_for_attempt(
        self,
        original_inputs: List[Dict],
        prompt_attempt_counts: Dict[str, int],
    ) -> Tuple[List[Dict], bool]:
        processed_inputs_for_attempt = copy.deepcopy(original_inputs)
        crop_smaller_than_bbox = False
        for i, item in enumerate(processed_inputs_for_attempt):
            attempt_num = prompt_attempt_counts.get(item["problem"], 0)

            pil_image = None
            if "image" in item and isinstance(item["image"], PIL.Image.Image):
                pil_image = item["image"].copy()
            elif "image_path" in item and item["image_path"] is not None:
                try:
                    pil_image = PIL.Image.open(item["image_path"]).convert("RGB")
                except Exception as e:
                    logger.warning(f"Could not load image for item {i} from {item['image_path']}: {e}")
                    item["image"] = None
                    continue
            else:
                logger.error(f"Item {i} has no valid image data")
                item["image"] = None
                continue

            if attempt_num > 0 and item.get("action") == "click":
                bbox_to_crop, _ = extract_bbox(item["solution"])
                if bbox_to_crop and pil_image:
                    try:
                        crop_results = cropping_utils.process_image_cropping(
                            pil_image, bbox_to_crop,
                            pil_image.width * (self.crop_factor ** attempt_num),
                            pil_image.height * (self.crop_factor ** attempt_num)
                        )
                        if crop_results:
                            crop_results = crop_results[0]
                            item["image"] = crop_results["cropped_pil_image"]
                            item["solution"] = modify_string_with_new_data(item["solution"], crop_results["new_bbox"])
                        else:
                            item["image"] = pil_image
                            crop_smaller_than_bbox = True
                    except Exception as e:
                        logger.error(f"Item {i} attempt {attempt_num}: Failed to crop: {e}. Using original.")
                        item["image"] = pil_image
                else:
                    item["image"] = pil_image
            else:
                item["image"] = pil_image
        
        return processed_inputs_for_attempt, crop_smaller_than_bbox


    def _generate_and_score_completions(
        self, inputs: List[Dict], model_to_generate_with: PreTrainedModel
    ) -> Dict[str, Any]:
        """
        Overrides parent method with minimal changes for prompt-wise cropping retry.
        """
        self._extract_bboxes_from_inputs(inputs) # Pre-process to get actions
        original_inputs = copy.deepcopy(inputs)
        prompt_attempt_counts = defaultdict(int)

        local_prompts = [x["problem"] for x in inputs]
        all_prompts_gathered = self.accelerator.gather_for_metrics(local_prompts)
        self.main_info("-"* 80)

        global_batch_size = len(all_prompts_gathered)
        if global_batch_size % self.num_generations != 0:
            raise ValueError("Global batch size is not divisible by num_generations. Prompt ordering cannot be determined.")
        num_unique_prompts = global_batch_size // self.num_generations
        unique_prompts_ordered = [all_prompts_gathered[i * self.num_generations] for i in range(num_unique_prompts)]

        for attempt_idx in range(self.max_crop_attempts):
            inputs_for_this_attempt, crop_smaller_than_bbox = self._prepare_images_for_attempt(
                original_inputs, prompt_attempt_counts
            )

            attempt_results_dict, attempt_metrics_dict, grouped_mean_rewards_maps = self._adapt_parent_generate_and_score(
                inputs_for_this_attempt, model_to_generate_with
            )
            # if crop_smaller_than_bbox:
            #     break

            if len(grouped_mean_rewards_maps) != len(unique_prompts_ordered):
                raise ValueError(f"Mismatch between number of unique prompts ({len(unique_prompts_ordered)}) and rewards ({len(grouped_mean_rewards_maps)}).")

            prompts_to_retry_count = 0
            
            for i, prompt_text in enumerate(unique_prompts_ordered):
                item_rewards = grouped_mean_rewards_maps[i]
                self.main_info(f"Rewards: {item_rewards} on device {self.accelerator.device}")
                grounding_reward = item_rewards.get(self.grounding_reward_key, -float('inf'))
                action_reward = item_rewards.get(self.action_reward_key, -float('inf'))

                is_successful = (grounding_reward > self.reward_threshold and action_reward > 0)
                

                if is_successful:
                    prompt_attempt_counts[prompt_text] = 0
                else:
                    current_attempts = prompt_attempt_counts.get(prompt_text, 0)
                    if current_attempts < self.max_crop_attempts -1 :
                        prompt_attempt_counts[prompt_text] += 1
                        prompts_to_retry_count += 1
                        # logger.info(f"Grounding Reward: {grounding_reward}, action reward: {action_reward}\nIncreasing attempt count to {prompt_attempt_counts[prompt_text]}.")
                        self.main_info(f"Grounding Reward: {grounding_reward}, action reward: {action_reward}\nIncreasing attempt to {prompt_attempt_counts[prompt_text]}.")
                    else:
                        # logger.warning(f"Grounding Reward: {grounding_reward}, action reward: {action_reward}\nfailed max attempts.")
                        self.main_warning(f"Grounding Reward: {grounding_reward}, action reward: {action_reward}\nfailed max attempts.")

            if prompts_to_retry_count == 0:
                # logger.info(f"All prompts succeeded or maxed out attempts. Finalizing at attempt {attempt_idx + 1}.\nRewards: {grouped_mean_rewards_maps}")
                self.main_info(f"All prompts succeeded or maxed out attempts. Finalizing at attempt {attempt_idx}.\nRewards: {grouped_mean_rewards_maps}")
                # self.attempt_num.append(attempt_idx + 1)  # Store the attempt number for this round
                self.attempt_num[f"{math.floor(self.state.epoch)}"].append(attempt_idx + 1)  # Store attempt number by epoch
                break
        
        # Log metrics from the very last attempt
        if attempt_metrics_dict:
            for key, val_list in attempt_metrics_dict.items():
                for v_item in val_list:
                    self._metrics[key].append(v_item)

        return attempt_results_dict