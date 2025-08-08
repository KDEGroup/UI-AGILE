from transformers import AutoProcessor
import torch
from PIL import Image
from vllm import SamplingParams, LLM
import numpy as np
import os
from loguru import logger
from qwen_vl_utils import process_vision_info





from typing import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from loguru import logger
from PIL import Image
from torch.nn import functional as F
class TransformersLogitsBasedScore:
    def __init__(self, model_path: str):
        self.logger = logger
        self.logger.info(f"Initializing model and processor from path: {model_path}")
        
        self.device = "cuda:1"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            # device_map=self.device,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        self._get_yes_token_id()

    def _get_yes_token_id(self):
        common_yes_tokens = ["Yes", " Yes", "yes", " yes"]
        self.yes_token_id = -1
        for token in common_yes_tokens:
            try:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id != self.processor.tokenizer.unk_token_id:
                    self.yes_token_id = token_id
                    self.logger.info(f"Found 'Yes' token '{token}' with ID: {self.yes_token_id}")
                    return
            except Exception:
                continue
        raise RuntimeError("Could not find a valid token ID for 'Yes'.")

    def calc_scores_for_batch(self, queries: List[str], images: List[Image.Image]) -> torch.Tensor:
        try:
            all_formatted_prompts = []
            for instruction in queries:
                content = f"Instruction: '{instruction}'. Question: Does this image accurately match the instruction? Yes or No? Answer:"
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": content}]}]
                formatted_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, padding_side='left'
                )
                all_formatted_prompts.append(formatted_prompt)

            inputs = self.processor(
                text=all_formatted_prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            first_token_logits = outputs.scores[0]
            log_probs = F.log_softmax(first_token_logits, dim=-1)
            scores = log_probs[:, self.yes_token_id]
            return scores.cpu()
        except Exception as e:
            self.logger.error(f"Error during score calculation: {e}")
            return torch.tensor([0.0] * len(images)).cpu()
        
if __name__ == "__main__":
    import json
    import os
    import time
    # Example usage of the LogitsBasedScore class
    model_path = "/path/to/your/model"
    score_calculator = TransformersLogitsBasedScore(model_path)
    time.sleep(20)