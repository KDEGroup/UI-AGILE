# main_updated.py

import os
import json
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import re
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from loguru import logger
from utils import extract_action, extract_coordinates
from eval import extract_param_value_loosely, gui_r1_extract_param

os.environ['TOKENIZERS_PARALLELISM'] = "false"
# --- Prompt Templates ---
# ac only have 7 examples with "long_press" action, so don't define it in the action space.
# ANDROID_CONTROL_DETAILED = """
# In this UI screenshot, I want to perform the command '{instruction}' with the action history '{history}'.
# Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait']) and the coordinate where the cursor is moved to(integer) if necessary.
# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
# The output answer format should be as follows:
# <think> ... </think> <answer>[{{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait'], 'coordinate': [x, y]}}]</answer>
# Note:
# Specific parameter (no default) is necessary for actions enum['input_text', 'open_app', 'scroll']
# Example:
# [{{'action': 'click', 'coordinate': [123, 300]}}]"
# [{{'action': 'input_text', 'input_text': 'hello world'}}]"
# [{{'action': 'open_app', 'app': 'edge'}}]"
# [{{'action': 'scroll', 'direction': enum['up', 'down', 'left', 'right']}}]
# [{{'action': 'navigate_back'}}]
# [{{'action': 'wait'}}]
# Please strictly follow the format.
# """

ANDROID_CONTROL_DETAILED = """
In this UI screenshot, I want to perform the command '{instruction}' with the action history '{history}'.
Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait']) and the coordinate where the cursor is moved to(integer) if necessary.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>[{{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait'], 'coordinate': [x, y]}}]</answer>
Note:
Specific parameter (no default) is necessary for actions enum['input_text', 'open_app', 'scroll'].
Swipe up is equivalent to scroll down, which usually means exploring more at the bottom of the page.
Example:
[{{'action': 'click', 'coordinate': [123, 300]}}]"
[{{'action': 'input_text', 'input_text': 'hello world'}}]"
[{{'action': 'open_app', 'app': 'edge'}}]"
[{{'action': 'scroll', 'direction': enum['up', 'down', 'left', 'right']}}]
[{{'action': 'navigate_back'}}]
[{{'action': 'wait'}}]
Please strictly follow the format.
"""

UI_R1_ANDROID_CONTROL = """
In this UI screenshot, I want to perform the command '{instruction}' with the action history '{history}'.
Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait']) and the coordinate where the cursor is moved to(integer) if necessary.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>[{{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait'], 'coordinate': [x, y]}}]</answer>
Please strictly follow the format.
"""

GUI_R1_ANDROID_CONTROL = """
You are RUN1-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{instruction}', with the action history being '{history}'.
Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:
<think> ... </think> [{{'action': enum['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}}]</answer>
Note:
specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll']
Example:
[{{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}}]
[{{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}}]
[{{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}}]
[{{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}}]
"""





# Add other templates if needed...
prompt_template_mapping = {
    "android_control_detailed": ANDROID_CONTROL_DETAILED,
    "ui_r1": UI_R1_ANDROID_CONTROL,
    "gui_r1": GUI_R1_ANDROID_CONTROL,
}


MODEL_TO_GT_ACTION_MAP = {
    "navigate_back": "press_back",
    "input_text": "type",

}

# --- Inference and Data Configuration ---
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=256, skip_special_tokens=False)
MICRO_BATCH = os.cpu_count()

CHUNK_SIZE = 800 # Adjust this based on your available RAM



class MultiModalDataset(Dataset):
    def __init__(self, data, processor, args=None):
        self.data = data
        self.processor = processor
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            image = Image.open(BytesIO(sample["image"]["bytes"]))
        except Exception as e:
            image = sample["image"]
        instruction = sample["instruction"]
        history = "None" if 'history' not in sample else sample['history']
        
        prompt_template = prompt_template_mapping.get(self.args.prompt_template)
        text_prompt = prompt_template.format(instruction=instruction, history=history)
        
        full_prompt = '<image>\n' + text_prompt
        message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": full_prompt}]}]

        prompt_for_model = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, _, video_kwargs = process_vision_info(message, return_video_kwargs=True)

        inputs = self.processor(text=[prompt_for_model], images=image_inputs, padding=True, return_tensors="pt")
        
        # Calculate scaling factors
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size
        origin_width, origin_height = image.size
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height
        # logger.debug(f"scale_x: {scale_x}, scale_y: {scale_y}")

        del inputs

        sample["scale"] = [scale_x.item(), scale_y.item()]
        sample["image_size"] = [origin_width, origin_height]

        mm_data = {"image": image_inputs}

        return {
            "prompt": prompt_for_model,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }

def custom_collate_fn(batch):
    # (Identical to original, no changes needed)
    collated_batch = {"prompts": [], "multi_modal_data": [], "mm_processor_kwargs": [], "original_samples": []}
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch

# --- VLLM Worker ---
class Worker:
    def __init__(self, model_path, sampling_params, args=None):
        self.args = args
        self.llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
        self.sampling_params = sampling_params

    def process_data(self, dataloader):
        results = []
        all_outputs = []
        all_original_samples = []
        for batch in tqdm(dataloader, desc="Running Inference"):
            llm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                for prompt, mm_data, mm_kwargs in zip(batch["prompts"], batch["multi_modal_data"], batch["mm_processor_kwargs"])
            ]
            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params, use_tqdm=False)
            all_outputs.extend(outputs)
            all_original_samples.extend(batch["original_samples"])

        for original_sample, output in zip(all_original_samples, all_outputs):
            generated_text = output.outputs[0].text
            # logger.debug(f"Generated Text: {generated_text} | GT Action: {original_sample.get('gt_action')}")

            # --- Extract and Normalize ---
            pred_action_raw = extract_action(generated_text)
            # Map model action to GT action for consistent evaluation
            pred_action_mapped = MODEL_TO_GT_ACTION_MAP.get(pred_action_raw, pred_action_raw)
            
            pred_coord, _, _ = extract_coordinates(generated_text)
            if pred_coord is None:
                pred_coord = [0, 0]
            scaled_coord = [pred_coord[0] * original_sample["scale"][0], pred_coord[1] * original_sample["scale"][1]]

            # --- Populate results ---
            result_sample = original_sample.copy()
            if not "gt_input_text" in result_sample:
                result_sample["gt_input_text"] = result_sample["gt_parameter"]
            result_sample["gt_input_text"] = result_sample["gt_input_text"].lower() if result_sample["gt_action"]=="scroll" else result_sample["gt_input_text"]
            result_sample["pred_raw"] = generated_text
            result_sample["pred_action"] = pred_action_mapped
            result_sample["pred_coord"] = scaled_coord
            result_sample["pred_input_text"] = extract_param_value_loosely(generated_text)
            if self.args.prompt_template == "gui_r1":
                result_sample["pred_input_text"] = gui_r1_extract_param(generated_text)
                logger.debug(f"GUI-R1 Pred Input Text: {result_sample['pred_input_text']} | GT Input Text: {result_sample['gt_input_text']}")
            # Clean up large fields for smaller output file
            result_sample["scale"] = []
            result_sample["image"] = ''
            results.append(result_sample)
        return results

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
    # Prepare output directory and file path
    output_dir = os.path.join(args.output_path, os.path.basename(args.model_path))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(args.data_path).replace(".jsonl", "_pred.jsonl").replace('.parquet', '_pred.jsonl'))
    logger.info(f"Output will be saved to: {output_file}")

    # Initialize components ONCE
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    worker = Worker(args.model_path, SAMPLING_PARAMS, args=args)

    # Open the output file to append results from each chunk
    with open(output_file, "w") as f:
        # Process data in chunks to save memory
        data_generator = yield_data_chunks(args.data_path, CHUNK_SIZE)
        for data_chunk in tqdm(data_generator, desc="Processing Chunks"):
            if not data_chunk:
                continue
            
            # Create dataset and dataloader for the current chunk
            dataset = MultiModalDataset(data_chunk, processor, args=args)
            dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
            
            # Run inference on the chunk
            chunk_results = worker.process_data(dataloader)
            
            # Write results for the current chunk to the file
            for sample in chunk_results:
                f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Inference complete and all results saved in {output_file}")
    return output_file



def calc_metrics(args):
    from eval import Evaluator
    log_file = args.output_path.replace('.jsonl', '.log')
    import logging
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Prevent duplicate logs if run multiple times
    logger.addHandler(logging.FileHandler(log_file, mode='w'))
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Prediction file: {os.path.basename(args.output_path)}")

    evaluator = Evaluator()
    evaluator.run(args.output_path)
    evaluator.report(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the VLLM model.")
    parser.add_argument('--prompt_template', type=str, default="uir1_think_point_detailed", choices=list(prompt_template_mapping.keys()))
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input data file (.jsonl or .parquet).")
    parser.add_argument('--output_path', type=str, default='./outputs', help="Directory to save prediction files.")
    args = parser.parse_args()
    args.output_path = main(args)
    calc_metrics(args)