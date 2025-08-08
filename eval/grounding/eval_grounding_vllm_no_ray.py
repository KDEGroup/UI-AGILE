import math
import os
from typing import *
import json
import traceback
from tqdm import tqdm
import argparse
import time
import uuid
import time
import concurrent.futures
import gc

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from qwen_vl_utils import process_vision_info

from loguru import logger
import numpy as np
import ast

from models.aguvis_constants import user_instruction, chat_template, grounding_system_message

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "train/src/trl_train/src"))
from open_r1.utils import extract_coordinates
from eval_screenspot_pro import evaluate, eval_sample_positive_gt
from gui_region.utils import process_image_cropping, get_element_sub_img
# from gui_region.Omniparser_Cropper.omniparser_cropper import BoundingBoxRefiner


NOTHINK_POINT = """
In this UI screenshot, I want to perform the command '{instruction}'.
Please provide the action to perform (enumerate in ['click']) and the coordinate where the cursor is moved to(integer) if click is performed.
Output the final answer in <answer> </answer> tags directly.
The output answer format should be as follows:
<answer>[{{'action': 'click', 'coordinate': [x, y]}}]</answer>
Please strictly follow the format.
"""

THINK_POINT = """
In this UI screenshot, I want to perform the command '{instruction}'.
Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text']) and the coordinate where the cursor is moved to(integer) if click is performed.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>[{{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text'], 'coordinate': [x, y]}}]</answer>
Please strictly follow the format.
"""




NOTHINK_BBOX = """
In this UI screenshot, I want to perform the command '{instruction}'.
Please provide the action to perform (enumerate in ['click']) and the 4d bounding box of target element(integer) rather than 2d coordinate if click is performed.
Output the final answer in <answer> </answer> tags directly.
The output answer format should be as follows:
<answer>[{{'action': 'click', 'bounding box': [x1, y1, x2, y2]}}]</answer>
Please strictly follow the format.
"""

THINK_BBOX = """
In this UI screenshot, I want to perform the command '{instruction}'.
Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text']) and the 4d bounding box of target element(integer) rather than 2d coordinate if click is performed.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>[{{'action': (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text']), 'bounding box': [x1, y1, x2, y2]}}]</answer>
Please strictly follow the format.
"""



GUI_R1 = """
You are RUN1-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{instruction}', with the action history being 'None'.
Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:
<think> ... </think> <answer>[{{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text [default]'}}]</answer>
Example:
[{{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}}]
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""


UGROUND = """
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {instruction}

Answer:"""

OS_ATLAS_BASE_7B = """
In this UI screenshot, what is the position of the element corresponding to the command \"{instruction}\" (with bbox)?
"""


Aguvis_7B_720P = user_instruction


# Can't reproduce showui's result. May have bugs.
SHOWUI_SYS = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."


prompt_template_mapping = {
    "nothink_point": NOTHINK_POINT,
    "think_point": THINK_POINT,
    "nothink_bbox": NOTHINK_BBOX,
    "think_bbox": THINK_BBOX,
    "uground": UGROUND,
    "os_atlas_base_7b": OS_ATLAS_BASE_7B,
    "aguvis_7b_720p" : Aguvis_7B_720P,
    "showui": SHOWUI_SYS,
    "gui_r1": GUI_R1,
    "uitars": GROUNDING_DOUBAO
}


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

CHUNK_SIZE = 600 # Adjust this based on your available RAM

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

# def smart_resize(
#     height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
# ) -> tuple[int, int]:
#     if min(height, width) == 0:
#         return 0, 0
#     if max(height, width) / min(height, width) > MAX_RATIO:
#         raise ValueError(
#             f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
#         )
#     h_bar = max(factor, round_by_factor(height, factor))
#     w_bar = max(factor, round_by_factor(width, factor))
#     if h_bar * w_bar > max_pixels:
#         beta = math.sqrt((height * width) / max_pixels)
#         h_bar = max(factor, floor_by_factor(height / beta, factor))
#         w_bar = max(factor, floor_by_factor(width / beta, factor))
#     elif h_bar * w_bar < min_pixels:
#         beta = math.sqrt(min_pixels / (height * width))
#         h_bar = ceil_by_factor(height * beta, factor)
#         w_bar = ceil_by_factor(width * beta, factor)
#     return h_bar, w_bar


class MultiModalDataset(Dataset):
    def __init__(self, data, processor, args):
        self.data = data
        self.processor = processor
        self.args = args

    def __len__(self):
        return len(self.data)

    def process_aguvis_input(instruction, image, processor, args):
        system_message = {
            "role": "system",
            "content": grounding_system_message
        }
        previous_actions = "None"
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": args.prompt_template.format(
                        overall_goal=instruction,
                        previous_actions=previous_actions,
                    ),
                }
            ],
        }
        recipient_text = "<|im_start|>assistant<|recipient|>os\n"
        messages = [system_message, user_message]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template
        )
        text += recipient_text
        image_data, _ = process_vision_info(messages)
        return text, image_data[0]


    def process_showui_input(instruction, image, processor, args):
        # logger.debug(f"show ui prompt_template ")
        min_pixels = 256*28*28
        max_pixels = 1344*28*28
        placeholder = [{"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels}]
        messages = [{
            "role": "system",
            "content": SHOWUI_SYS
        }, {
            "role":
            "user",
            "content": [
                *placeholder,
                {
                    "type": "text",
                    "text": instruction
                },
            ],
        }]
        processed_text = processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)
        return processed_text, image_data[0]


    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample["instruction"]
        image_bytes = sample["image"]["bytes"]
        gt_bbox = sample.get("bbox", None)
        image = Image.open(BytesIO(image_bytes))
        # SE-GUI have low max_pixels in preprocessor_config, causing scale to be about 2.
        resized_height, resized_width = smart_resize(image.height, image.width,
                                                     factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                                                     min_pixels=self.processor.image_processor.min_pixels,
                                                     max_pixels=self.processor.image_processor.max_pixels)
        scale_x = image.width / resized_width if resized_width > 0 else 1
        scale_y = image.height / resized_height if resized_height > 0 else 1
        # logger.debug(f"scale_x: {scale_x}, scale_y: {scale_y}")
        # customerize for aguvis
        if self.args.prompt_template == Aguvis_7B_720P:
            processed_text, image_data = MultiModalDataset.process_aguvis_input(instruction, image, self.processor, args)
        # customerize for showui
        elif self.args.prompt_template == SHOWUI_SYS:
            processed_text, image_data = MultiModalDataset.process_showui_input(instruction, image, self.processor, args)
        else:
            prompt = self.args.prompt_template.format(instruction=instruction)
            query = f"<image>\n{prompt}"
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]}]
            processed_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)
            image_data = image_data[0]
        # logger.debug(f"Processed text for sample {idx}: {processed_text}")
        return {
            "instruction": instruction,
            "processed_text": processed_text,
            "image": image,
            "image_data": image_data,
            "scale": (scale_x, scale_y),
            "bbox": gt_bbox if gt_bbox else None,
        }


    def process_single_input(instruction, image, processor, args):
        """
        Process a single input image and instruction based on the specified prompt template.
        Returns the processed text and image data.
        """
        resized_height, resized_width = smart_resize(image.height, image.width,
                                                     factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
                                                     min_pixels=processor.image_processor.min_pixels,
                                                     max_pixels=processor.image_processor.max_pixels)
        scale_x = image.width / resized_width if resized_width > 0 else 1
        scale_y = image.height / resized_height if resized_height > 0 else 1
        if args.prompt_template == Aguvis_7B_720P:
            return *MultiModalDataset.process_aguvis_input(instruction, image, processor, args), (scale_x, scale_y)
        elif args.prompt_template == SHOWUI_SYS:
            return *MultiModalDataset.process_showui_input(instruction, image, processor, args), (scale_x, scale_y)
        else:
            prompt = args.prompt_template.format(instruction=instruction)
            query = f"<image>\n{prompt}"
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]}]
            processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)
            return processed_text, image_data[0], (scale_x, scale_y)



def custom_collate_fn(batch):
    instructions = [item['instruction'] for item in batch]
    processed_texts = [item['processed_text'] for item in batch]
    images = [item['image'] for item in batch]
    image_data = [item['image_data'] for item in batch]
    scales = [item['scale'] for item in batch]
    bboxes = [item['bbox'] for item in batch]
    return {
        "instructions": instructions,
        "processed_texts": processed_texts,
        "images": images,
        "image_data": image_data,
        "scales": scales,
        "bboxes": bboxes,
    }

class VLLMWorker:
    def __init__(self, model_path, args=None):
        logger.info("Initializing VLLMWorker...")
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False)
        self.model = LLM(model=model_path, tensor_parallel_size=args.num_gpus_for_generate, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.args = args
        logger.info("VLLMWorker initialized.")

    def process_data_and_predict(self, dataloader):
        prompts = []
        scales = []
        bboxes = []
        logger.info("Aggregating data from the dataloader...")
        for batch in tqdm(dataloader, desc="Preparing batches"):
            for i in range(len(batch["processed_texts"])):
                prompts.append({
                    "prompt": batch["processed_texts"][i],
                    "multi_modal_data": {"image": batch["image_data"][i]}
                })
                scales.append(batch["scales"][i])
                bboxes.append(batch["bboxes"][i])
        if not prompts:
            return []
        logger.info(f"Starting vLLM batch inference for {len(prompts)} samples...")
        # outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=True)
        outputs = []
        begin_time = time.time()
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=True)
        # batch_size = CHUNK_SIZE
        # for i in tqdm(range(0, len(prompts), batch_size), desc=f"batch inference {batch_size}"):
        #     outputs.extend(self.model.generate(prompts[i:i + batch_size], self.sampling_params))
        logger.info(f"Batch inference completed in {time.time() - begin_time:.2f} seconds.")
        results = []
        for i, output in enumerate(tqdm(outputs, desc="Processing results")):
            response = output.outputs[0].text
            # logger.debug(f"Response {i}: {response}")
            scale_x, scale_y = scales[i]
            if self.args.prompt_template == SHOWUI_SYS:
                try:
                    pred_coord = ast.literal_eval(response)
                except Exception as e:
                    logger.error(f"Error parsing response: {response} with error {e}")
                    pred_coord = [0, 0]
            else:
                pred_coord, _, _ = extract_coordinates(response)
            point = []
            pred_bbox = None
            if not pred_coord:
                logger.error(f"Can't' extract_coordinates from {response}")
            elif pred_coord:
                # for qwen2vl based model that pred range is in [0, 1000]. Map to w, h
                if self.args.model_type == "qwen2vl":
                    w, h = prompts[i]["multi_modal_data"]["image"].size
                    pred_coord = [c / 1000 for c in pred_coord]
                    if len(pred_coord) == 2:
                        pred_coord = [pred_coord[0] * w, pred_coord[1] * h]
                    elif len(pred_coord) == 4:
                        pred_coord = [pred_coord[0] * w, pred_coord[1] * h,
                                     pred_coord[2] * w, pred_coord[3] * h]
                elif self.args.model_type == "aguvis_7b_720p":
                    # aguvis model output is in [0, 1] range
                    w, h = prompts[i]["multi_modal_data"]["image"].size
                    if len(pred_coord) == 2:
                        pred_coord = [pred_coord[0] * w, pred_coord[1] * h]
                    elif len(pred_coord) == 4:
                        pred_coord = [pred_coord[0] * w, pred_coord[1] * h,
                                      pred_coord[2] * w, pred_coord[3] * h]
                if len(pred_coord) == 2:
                    point = [pred_coord[0] * scale_x, pred_coord[1] * scale_y]
                elif len(pred_coord) == 4:
                    pred_bbox = [pred_coord[0] * scale_x, pred_coord[1] * scale_y,
                                  pred_coord[2] * scale_x, pred_coord[3] * scale_y]
                    point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]

            correctness = "wrong"
            gt_bbox = bboxes[i]
            if gt_bbox and point:
                if point[0] >= gt_bbox[0] and point[0] <= gt_bbox[2] and point[1] >= gt_bbox[1] and point[1] <= gt_bbox[3]:
                    correctness = "correct"
            results.append({
                "raw_response": response,
                "point": [float(c) for c in point] if point else None,
                "pred_bbox": pred_bbox,
                "correctness": correctness,
                "score_eval_data": None,
            })
        return results





# speedup the import process
# vllm fully use
# only for lgoits based score
import traceback
import uuid
from collections import defaultdict

def resize_image(image: Image.Image, min_size: int = 28) -> Image.Image:
    """
    Make sure the image w,h bigger than min_size while maintaining aspect ratio.
    And avoid ValueError: absolute aspect ratio must be smaller than 200, got xxx
    """
    w, h = image.size
    short_side = min(w, h)
    
    if short_side >= min_size:
        return image
        
    scale = min_size / short_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Note: The resize operation itself might raise a ValueError if the original
    # aspect ratio is too extreme (>200 or <1/200). A safer approach is to
    # check and crop the image *before* attempting to resize it.
    image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # The following block handles cases where the aspect ratio is extreme
    # by cropping the image to a maximum 200:1 or 1:200 ratio.
    if new_w / new_h > 200:
        # Image is excessively wide, crop horizontally
        target_w = int(new_h * 200)
        left = (new_w - target_w) // 2
        right = left + target_w
        image = image.crop((left, 0, right, new_h))
    elif new_h / new_w > 200:
        # Image is excessively tall, crop vertically
        target_h = int(new_w * 200)
        top = (new_h - target_h) // 2
        bottom = top + target_h
        image = image.crop((0, top, new_w, bottom))
        
    return image

class CropSelectVLLMWorkerV2:
    def __init__(self, model_path, args=None):
        logger.info("Initializing Refactored CropSelectVLLMWorker...")
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False)
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.args = args
        self.image_save_dir = self.args.score_eval_image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)

        from gui_region.logits_based_score import TransformersLogitsBasedScore
        self.score_object = TransformersLogitsBasedScore(self.args.score_model_path)

        self.model = LLM(model=self.model_path, tensor_parallel_size=1, trust_remote_code=True)
        # self.bbox_refiner = BoundingBoxRefiner()

        
        logger.info("CropSelectVLLMWorker initialized.")


    def process_data_and_predict(self, dataloader):
        """
        Orchestrates the refactored, two-phase batch processing workflow.
        """
        # --- Phase 1: Batch predict coordinates for all tiles from all samples ---
        # This phase generates all possible candidate predictions across the entire dataset.
        logger.info("[Phase 1/3] Generating all candidate coordinates...")
        phase1_results = self._phase1_batch_predict_coordinates(dataloader)
        logger.info(f"[Phase 1/3] Completed. Found potential candidates for {len(phase1_results)} samples.")

        # --- Phase 2: Batch score all candidates and select the best one per sample ---
        # This phase takes all candidates, scores them in a single batch, and selects the best.
        logger.info("[Phase 2/3] Scoring all candidates...")
        final_predictions, images_to_save = self._phase2_batch_score_candidates(phase1_results, dataloader)
        logger.info(f"[Phase 2/3] Completed. Final predictions generated for {len(final_predictions)} samples.")

        # --- Phase 3: Batch save images for evaluation ---
        # This phase handles all disk I/O at the end to avoid blocking compute.
        logger.info("[Phase 3/3] Saving images for evaluation...")
        self._save_evaluation_images(images_to_save)
        logger.info(f"[Phase 3/3] Completed. Saved {len(images_to_save)} images.")

        return final_predictions


    def _phase1_task_generator(self, dataloader):
        for sample_idx, batch in enumerate(dataloader):
            for i in range(len(batch["images"])):
                global_idx = sample_idx * dataloader.batch_size + i
                yield (
                    global_idx,
                    batch["images"][i],
                    batch["instructions"][i],
                    batch["bboxes"][i],
                    self.args,
                    self.processor
                )


    def _phase1_process_single_sample(self, args_tuple):
        global_idx, image, instruction, gt_bbox, args, processor = args_tuple
        output_tile_data = process_image_cropping(
            image,
            step_x=int(image.width * args.step_ratio),
            step_y=int(image.height * args.step_ratio),
            tile_width=int(image.width * args.tile_ratio),
            tile_height=int(image.height * args.tile_ratio)
        )

        if not output_tile_data:
            return [], []

        sample_prompts = []
        sample_metadata = []
        for tile_img, (crop_x_min, crop_y_min) in output_tile_data:
            processed_text, tile_image_data, scale = MultiModalDataset.process_single_input(
                instruction, tile_img, processor, args
            )
            sample_prompts.append({
                "prompt": processed_text,
                "multi_modal_data": {"image": tile_image_data}
            })
            sample_metadata.append({
                "global_idx": global_idx,
                "instruction": instruction,
                "gt_bbox": gt_bbox,
                "full_image": image,
                "tile_img": tile_img,
                "scale": scale,
                "crop_offset": (crop_x_min, crop_y_min)
            })
            
        return sample_prompts, sample_metadata

    def _phase1_batch_predict_coordinates(self, dataloader):
        total_samples = len(dataloader.dataset)
        t1 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            task_gen = self._phase1_task_generator(dataloader)
            results = list(tqdm(
                executor.map(self._phase1_process_single_sample, task_gen),
                total=total_samples,
                desc="Phase 1.1: Processing Samples (Pipeline)"
            ))
        logger.info(f"Phase 1.1 Processing inputs completed in {time.time() - t1:.2f} seconds.")
        vllm_prompts = []
        tile_metadata = []
        for prompts_for_sample, metadata_for_sample in results:
            vllm_prompts.extend(prompts_for_sample)
            tile_metadata.extend(metadata_for_sample)

        if not vllm_prompts:
            return []


        outputs = []
        t1 = time.time()
        outputs = self.model.generate(vllm_prompts, self.sampling_params, use_tqdm=True)
        logger.info(f"Batch inference completed in {time.time() - t1:.2f} seconds.")

        # Process results and group candidates by original sample index
        # Using defaultdict to easily append candidates to each sample
        candidates_by_sample = defaultdict(list)
        for i, output in enumerate(tqdm(outputs, desc="Phase 1: Processing Tile Results")):
            metadata = tile_metadata[i]
            scale = metadata["scale"]
            response = output.outputs[0].text
            if self.args.prompt_template == SHOWUI_SYS:
                try:
                    pred_coord_local = ast.literal_eval(response)
                except Exception as e:
                    logger.error(f"Error parsing response: {response} with error {e}")
                    pred_coord_local = [0, 0]
            else:
                pred_coord_local, _, _ = extract_coordinates(response)

            if not pred_coord_local:
                logger.error(f"Can't' extract_coordinates from {response}")
                pred_coord_local = [0, 0]  # Default to (0, 0) if extraction fails
            # handle qwen resize scale
            pred_coord_local = [x * scale[0] if i % 2==0 else x * scale[1]  for i, x in enumerate(pred_coord_local)]
            # for qwen2vl based model that pred range is in [0, 1000]. Map to w, h
            if self.args.model_type == "qwen2vl":
                w, h = metadata["tile_img"].size
                pred_coord_local = [c / 1000 for c in pred_coord_local]
                pred_coord_local = [x * w if i % 2==0 else x * h for i, x in enumerate(pred_coord_local)]
            elif self.args.model_type == "aguvis_7b_720p":
                # aguvis model output is in [0, 1] range
                w, h = metadata["tile_img"].size
                pred_coord_local = [x * w if i % 2==0 else x * h for i, x in enumerate(pred_coord_local)]
            # This candidate is valid, add it to the list for its original sample
            candidates_by_sample[metadata["global_idx"]].append({
                "response": response,
                "local_coord": pred_coord_local,
                **metadata # Pass all metadata to the next phase
            })
        
        # Convert defaultdict to a regular list of dicts for the next phase
        # Each item in the list corresponds to one original sample and contains all its candidates
        phase1_results = []
        for global_idx, candidates in candidates_by_sample.items():
            # We just need one instance of the metadata for the original sample
            sample_info = {
                "global_idx": global_idx,
                "instruction": candidates[0]["instruction"],
                "gt_bbox": candidates[0]["gt_bbox"],
                "full_image": candidates[0]["full_image"],
                "candidates": candidates # List of all valid predictions for this sample
            }
            phase1_results.append(sample_info)
        return phase1_results



    def _prepare_candidate_inputs(self, phase1_results: List[Dict]) -> Tuple[List[List], List[Dict]]:
        all_ele_inputs = []
        all_ele_metadata = []
        for sample_idx, sample in enumerate(phase1_results):
            for candidate_idx, candidate in enumerate(sample["candidates"]):
                all_ele_inputs.append(['', candidate["tile_img"], candidate["local_coord"]])
                all_ele_metadata.append({
                    "sample_idx": sample_idx,
                    "candidate_idx_in_sample": candidate_idx,
                    "sample_global_idx": sample["global_idx"],
                    "instruction": sample["instruction"],
                    "gt_bbox": sample["gt_bbox"],
                    "candidate_info": candidate,
                })
        return all_ele_inputs, all_ele_metadata

    def _batch_refine_candidates(self, all_ele_inputs: List[List], refiner_batch_size: int = 150) -> List[Image.Image]:
        logger.info(f"Refining {len(all_ele_inputs)} bounding boxes in batches of {refiner_batch_size}...")
        all_element_imgs = []
        t1 = time.time()
        for i in tqdm(range(0, len(all_ele_inputs), refiner_batch_size), desc="Refining BBoxes"):
            batch_inputs = all_ele_inputs[i:i + refiner_batch_size]
            try:
                batch_element_imgs = self.bbox_refiner.get_element_img_batch(batch_inputs)
                all_element_imgs.extend(batch_element_imgs)
            except Exception as e:
                logger.error(f"Error in bbox_refiner batch starting at index {i}: {e}")
                logger.debug(traceback.format_exc())
                all_element_imgs.extend([None] * len(batch_inputs))
        logger.info(f"Refinement completed in {time.time() - t1:.2f} seconds.")
        return all_element_imgs

    def _process_refined_candidates(self, all_element_imgs: List[Image.Image], all_ele_metadata: List[Dict]) -> Tuple[List, List, List, Dict, List]:
        scoring_instructions, scoring_candidate_imgs, scoring_metadata, images_to_save = [], [], [], []
        sample_eval_info = defaultdict(lambda: {"candidate_imgs_paths": [], "correct_prediction_indices": []})

        for i, element_img in enumerate(tqdm(all_element_imgs, desc="Processing refined candidates")):
            if element_img is None: continue
            
            meta = all_ele_metadata[i]
            try:
                element_img_to_score = resize_image(element_img, min_size=IMAGE_FACTOR)
                
                scoring_instructions.append(meta["instruction"])
                scoring_candidate_imgs.append(element_img_to_score)

                img_name = f"sample_{meta['sample_global_idx']}_tile_{meta['candidate_idx_in_sample']}.png"
                images_to_save.append({"image": element_img_to_score, "path": os.path.join(self.image_save_dir, img_name)})
                sample_eval_info[meta['sample_idx']]["candidate_imgs_paths"].append(img_name)

                candidate = meta["candidate_info"]
                crop_x_min, crop_y_min = candidate["crop_offset"]
                pred_coord_local = candidate["local_coord"]
                response_for_eval = {}
                if len(pred_coord_local) == 4:
                    global_coord = [pred_coord_local[0] + crop_x_min, pred_coord_local[1] + crop_y_min, pred_coord_local[2] + crop_x_min, pred_coord_local[3] + crop_y_min]
                    response_for_eval["bbox"] = global_coord
                else:
                    global_coord = [pred_coord_local[0] + crop_x_min, pred_coord_local[1] + crop_y_min]
                    response_for_eval["point"] = global_coord
                
                scoring_metadata.append({"global_idx": meta["sample_global_idx"], "global_coord": global_coord, "raw_response": candidate["response"]})

                if meta["gt_bbox"] and eval_sample_positive_gt({"bbox": meta["gt_bbox"]}, response_for_eval) == "correct":
                    sample_eval_info[meta['sample_idx']]["correct_prediction_indices"].append(meta['candidate_idx_in_sample'])

            except Exception as e:
                logger.error(traceback.format_exc())

        return scoring_instructions, scoring_candidate_imgs, scoring_metadata, sample_eval_info, images_to_save


    def _score_and_select_best_candidate(self, phase1_results, dataloader, scoring_instructions, scoring_candidate_imgs, scoring_metadata, score_batch_size=150) -> Dict:
        logger.info(f"Performing batch scoring on {len(scoring_candidate_imgs)} candidate elements...")
        all_scores = []
        t1 = time.time()
        if len(scoring_candidate_imgs) > score_batch_size:
            for i in tqdm(range(0, len(scoring_candidate_imgs), score_batch_size), desc="Scoring Candidates"):
                batch_scores = self.score_object.calc_scores_for_batch(scoring_instructions[i:i+score_batch_size], scoring_candidate_imgs[i:i+score_batch_size])
                all_scores.extend(batch_scores)
        else:
            all_scores = self.score_object.calc_scores_for_batch(scoring_instructions, scoring_candidate_imgs)
        logger.info(f"Scoring completed in {time.time() - t1:.2f} seconds.")
        scores_by_sample = defaultdict(list)
        for i, meta in enumerate(scoring_metadata):
            scores_by_sample[meta["global_idx"]].append({
                "score": all_scores[i].item(), "global_coord": meta["global_coord"], "raw_response": meta["raw_response"],
            })
        
        final_predictions = {}
        for sample in phase1_results:
            global_idx = sample["global_idx"]
            candidates = scores_by_sample.get(global_idx)
            if not candidates:
                final_predictions[global_idx] = self._create_default_result(sample)
                continue
            
            best_candidate = max(candidates, key=lambda x: x['score'])
            best_pred_coord = best_candidate["global_coord"]
            point, pred_bbox = (None, None)
            
            if len(best_pred_coord) == 4:
                pred_bbox = best_pred_coord
                point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
            else:
                point = best_pred_coord
            
            correctness = "wrong"
            if sample["gt_bbox"] and point and (sample["gt_bbox"][0] <= point[0] <= sample["gt_bbox"][2]) and (sample["gt_bbox"][1] <= point[1] <= sample["gt_bbox"][3]):
                correctness = "correct"
            
            final_predictions[global_idx] = {
                "raw_response": best_candidate["raw_response"], "point": [float(c) for c in point] if point else None,
                "pred_bbox": [float(c) for c in pred_bbox] if pred_bbox else None, "correctness": correctness,
                "n_pass_correctness": sample.get("n_pass_correctness", "wrong"),
                "score_eval_data": sample.get("score_eval_data", None),
            }
        return final_predictions



    def _phase2_batch_score_candidates(self, phase1_results, dataloader, score_batch_size=8):
        all_ele_inputs, all_ele_metadata = self._prepare_candidate_inputs(phase1_results)
        if not all_ele_inputs:
            return [self._create_default_result(s) for s in phase1_results], []
        # all_element_imgs = self._batch_refine_candidates(all_ele_inputs)
        all_element_imgs = []
        for sample_idx, sample in enumerate(phase1_results):
            for candidate_idx, candidate in enumerate(sample["candidates"]):
                try:
                    element_img = get_element_sub_img(candidate["tile_img"], candidate["local_coord"])
                    all_element_imgs.append(element_img)
                except Exception as e:
                    logger.error(traceback.format_exc())
                    all_element_imgs.append(candidate["tile_img"])
        scoring_instructions, scoring_candidate_imgs, scoring_metadata, sample_eval_info, images_to_save = \
            self._process_refined_candidates(all_element_imgs, all_ele_metadata)
        
        for i, sample in enumerate(phase1_results):
            eval_info = sample_eval_info.get(i)
            if eval_info and eval_info["correct_prediction_indices"]:
                sample["n_pass_correctness"] = "correct"
                gt_img_name = f"sample_{sample['global_idx']}_gt.png"
                gt_img_path = os.path.join(self.image_save_dir, gt_img_name)
                images_to_save.append({"image": sample["full_image"].crop(sample["gt_bbox"]), "path": gt_img_path})
                sample["score_eval_data"] = {
                    "instruction": sample["instruction"], "image_paths": eval_info["candidate_imgs_paths"],
                    "correct_prediction_img_indices": eval_info["correct_prediction_indices"], "groundtruth_image": gt_img_path,
                }
            else:
                sample["n_pass_correctness"] = "wrong"

        if not scoring_candidate_imgs:
            return [self._create_default_result(s) for s in phase1_results], images_to_save
            
        final_predictions_map = self._score_and_select_best_candidate(
            phase1_results, dataloader, scoring_instructions, scoring_candidate_imgs, scoring_metadata, score_batch_size
        )

        num_total_samples = sum(len(b['images']) for b in dataloader)
        all_final_results = [final_predictions_map.get(i, self._create_default_result()) for i in range(num_total_samples)]
        del scoring_candidate_imgs        
        del dataloader
        return all_final_results, images_to_save


    def _create_default_result(self, sample_info=None):
        """Creates a default result for samples with no valid predictions."""
        result = {
            "raw_response": "No valid coordinates predicted.",
            "point": None,
            "pred_bbox": None,
            "correctness": "wrong",
            "n_pass_correctness": "wrong",
            "score_eval_data": None
        }
        if sample_info:
            result["n_pass_correctness"] = sample_info.get("n_pass_correctness", "wrong")
            result["score_eval_data"] = sample_info.get("score_eval_data", None)
        return result
    
    def _save_evaluation_images(self, images_to_save):
        """Saves all collected images to disk in a single batch."""
        if not images_to_save:
            return
        for item in tqdm(images_to_save, desc="Phase 3: Saving Images"):
            try:
                item["image"].save(item["path"])
            except Exception as e:
                logger.error(f"Failed to save image to {item['path']}: {e}")





def yield_data_chunks(data_path, chunk_size):
    """
    Reads data from a .jsonl or .parquet file and yields it in chunks
    to avoid loading the entire file into memory.
    """
    if data_path.endswith('.parquet'):
        # The 'datasets' library can stream, which is memory-efficient.
        dataset = load_dataset("parquet", data_files=data_path, split="train", streaming=True)
        chunk = []
        for sample in dataset:
            chunk.append(sample)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk # Yield the last, smaller chunk
    else: # Assumes .jsonl
        with open(data_path, "r") as f:
            chunk = []
            for line in f:
                chunk.append(json.loads(line))
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk # Yield the last, smaller chunk


def main(args):

    if args.crop_select:
        logger.info(f"Initializing score object with method: {args.score_method}")
    else:
         logger.info(f"  - Scoring Model is used.")

    args.prompt_template = prompt_template_mapping.get(args.prompt_template)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if args.crop_select:
        worker = CropSelectVLLMWorkerV2(args.model_path, args)
    else:
        worker = VLLMWorker(args.model_path, args)

    all_results = []
    all_dataset = []
    data_generator = yield_data_chunks(args.data_path, CHUNK_SIZE)
    for data_chunk in tqdm(data_generator, desc="Processing Chunks"):
        dataset = data_chunk
        torch_dataset = MultiModalDataset(dataset, processor, args)
        dataloader = DataLoader(
            torch_dataset, 
            num_workers=os.cpu_count(), collate_fn=custom_collate_fn
        )
        logger.info(f"Processing chunk with {len(dataset)} samples...")
        results = worker.process_data_and_predict(dataloader)
        logger.info(f"All {len(results)} results received.")
        all_results.extend(results)
        for d in dataset:
            d.pop('image', None)  # Remove image to save memory
        all_dataset.extend(dataset)
        gc.collect()
    # logger.debug(f"Total results collected: {all_results}")
    # logger.debug(f"Total dataset samples collected: {all_dataset}")
    final_results = []
    all_score_eval_data = []
    for i in tqdm(range(len(all_results)), desc="Merging results"):
        sample_result = {**all_dataset[i], **all_results[i]}
        sample_result.pop("image", None)
        score_eval_data = sample_result.pop("score_eval_data", None)
        if score_eval_data:
            all_score_eval_data.append(score_eval_data)
        final_results.append(sample_result)

    if args.crop_select and all_score_eval_data:
        logger.info(f"Saving score evaluation data to {args.score_eval_out_path}")
        with open(args.score_eval_out_path, 'w') as f:
            json.dump(all_score_eval_data, f, indent=4)

    if final_results:
        for result in final_results:
            result["platform"] = result.get("data_source", None) if not result.get("platform") else result["platform"]
            result["ui_type"] = result.get("data_type", None) if not result.get("ui_type") else result["ui_type"]
            result["application"] = result.get("data_source", None) if not result.get("application") else result["application"]
        
        report = evaluate(final_results)      
        if args.crop_select:
            n_pass_accuracy = sum(1 for r in final_results if r.get('n_pass_correctness') == 'correct') / len(final_results)
            logger.info(f"N pass accuracy: {n_pass_accuracy:.4f}")
            report["n_pass_accuracy"] = n_pass_accuracy       
        try:
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
            with open(args.log_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Evaluation of ScreenSpot finished. Report saved to {args.log_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning("No results were generated, skipping evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the VLLM model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file (parquet format).")
    parser.add_argument("--log_path", type=str, default="logs/evaluation_report_single_gpu.json", help="Path to save the evaluation report.")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader.")
    parser.add_argument('--step_ratio', type=float, default=0.5, help="Step ratio for cropping.")
    parser.add_argument('--tile_ratio', type=float, default=0.6, help="Tile size for cropping.")
    parser.add_argument('--model_type', type=str, default="qwen2.5vl", choices=["qwen2.5vl", "qwen2vl", "aguvis_7b_720p"])
    parser.add_argument('--prompt_template', type=str, default="uir1_nothink_point", choices=list(prompt_template_mapping.keys()), help="Prompt template to use for the model.")
    parser.add_argument('--num_gpus_for_generate', type=int, default=1, help="Number of GPUs to use for vLLM tensor parallelism.")
    parser.add_argument('--crop_select', action='store_true', help="Use crop-and-select strategy for grounding.")
    parser.add_argument('--score_method', default="dual_encoder", choices=["clip", "colqwen", "dual_encoder", "lmm_logits"], help="Method to calculate scores for cropping.")
    parser.add_argument('--score_model_path', type=str, default="gui_region/guiclip", help="Path to the model used for scoring.")
    parser.add_argument('--score_eval_image_save_dir', type=str, default="score_eval_images", help="Directory to save images for score evaluation.")
    parser.add_argument('--score_eval_out_path', type=str, default="score_eval_data.json", help="Path to save the score evaluation data.")
    
    args = parser.parse_args()
    main(args)