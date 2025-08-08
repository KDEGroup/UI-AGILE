# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import PIL

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerWithCropping, Qwen2VLGRPOTrainerWithCropping

from open_r1.utils import *
from open_r1.reward import action_reward, grounding_reward, format_reward_uir1_wrapper, simple_length_reward, plain_length_reward, parameter_reward, binary_grounding_reward
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import math
from loguru import logger

THINK_POINT_DETAILED = """
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



NOTHINK_POINT_DETAILED = """
In this UI screenshot, I want to perform the command '{instruction}' with the action history '{history}'.
Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait']) and the coordinate where the cursor is moved to(integer) if necessary.
Output the final answer in <answer> </answer> tags directly. 
The output answer format should be as follows:
<answer>[{{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text', 'wait'], 'coordinate': [x, y]}}]</answer>
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


UI_R1 = """
In this UI screenshot, I want to perform the command '{instruction}'.
Please provide the action to perform (enumerate in ['click']) and the coordinate where the cursor is moved to(integer) if necessary.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<answer>[{{'action': enum['click'], 'coordinate': [x, y]}}]</answer>
Please strictly follow the format.
"""

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    resample_if_useless: bool = field(
        default=True,
        metadata={"help": "resample_if_useless"}
    )
    thinking_strategy: str = field(
        default="dast",
        metadata={"help": "Thinking strategy to use. Possible values: 'nothinking', 'dast', 'simple_thinking', 'simple_thinkingv2'"},
    )
    binary_grounding_reward: bool = field(
        default=False,
        metadata={"help": "Use binary grounding reward instead of grounding reward."},
    )
    grounding_only: bool = field(
        default=False
    )
    pred_type: str = field(
        default="coord",
        metadata={"help": "Prediction type to use. Possible values: 'coord', 'bbox'"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["action_reward", "grounding_reward", "format_reward_uir1"],
        metadata={"help": "List of reward functions."},
    )    
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )








@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False



def main(script_args, training_args, model_args):
    # Get reward functions

    reward_funcs_registry = {
        "action_reward": action_reward,
        "format_reward_uir1": format_reward_uir1_wrapper(script_args.thinking_strategy, script_args.pred_type),
        "simple_length_reward": simple_length_reward,
        "plain_length_reward": plain_length_reward,
        "parameter_reward" : parameter_reward
    }
    if script_args.binary_grounding_reward:
        reward_funcs_registry["grounding_reward"] = binary_grounding_reward
    else:
        reward_funcs_registry["grounding_reward"] = grounding_reward
    script_args.reward_funcs = ['action_reward','grounding_reward','format_reward_uir1', 'parameter_reward']
    if script_args.thinking_strategy == "simple_thinking":
        script_args.reward_funcs.append('simple_length_reward')
    elif script_args.thinking_strategy == "simple_thinkingv2":
        script_args.reward_funcs.append('plain_length_reward')
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            # for line in f:
            data = json.load(f)
            for item in data:
                if 'img_filename' in item:
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(image_folder, item['img_filename'])
                    del item['img_filename'] # remove the image column so that it can be loaded later

                if not script_args.grounding_only and script_args.pred_type == "coord":
                    if script_args.thinking_strategy == "nothinking":
                        logger.warning("override problem for more dataset using detailed prompt")
                        item['problem'] = NOTHINK_POINT_DETAILED.format(
                            instruction=item['instruction'],
                            history=item.get('history', 'None')
                        )
                    else:
                        item['problem'] = THINK_POINT_DETAILED.format(
                            instruction=item['instruction'],
                            history=item.get('history', 'None')
                        ) 
                elif script_args.grounding_only and script_args.pred_type == "coord":
                    item['problem'] = UI_R1.format(
                        instruction=item['instruction']
                    )

                item['parameter'] = item.get('parameter', None)
                if item['action']=="click" and ('bbox' in item or 'gt_bbox' in item):
                    coordinate_value = item.get("bbox", item.get("gt_bbox"))
                    item['solution'] = f"<answer>[{{'action': 'click' ,'coordinate': {coordinate_value} }}]</answer>"
                else:
                    item['solution'] = f"<answer>[{{'action': '{item['action']}' ,'parameter': {item.get('parameter')}}}]</answer>"
                
                all_data.append(item)

    dataset = Dataset.from_list(all_data)
    # dataset = dataset.shuffle()
    def make_conversation_from_json(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                'image_path': example['image_path'],  # Store path instead of loaded image
                'solution': example['solution'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }],
            }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
    dataset = dataset.map(make_conversation_from_json, num_proc=8)

    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    # if script_args.resample_if_useless:
    #     trainer_cls = Qwen2VLGRPOTrainerResampleIfUseless
    if not training_args.use_vllm:
        if script_args.resample_if_useless:
            trainer_cls = Qwen2VLGRPOTrainerWithCropping
        else:
            trainer_cls = Qwen2VLGRPOTrainer
    else:
        if script_args.resample_if_useless:
            trainer_cls = Qwen2VLGRPOVLLMTrainerWithCropping
        else:
            trainer_cls = Qwen2VLGRPOVLLMTrainer
    logger.info(f"using: {trainer_cls}")


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        thinking_strategy=script_args.thinking_strategy,
        grounding_reward_key=f'rewards/{reward_funcs_registry["grounding_reward"].__name__}',
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # Store the original item that failed to generate properly and attempt number
    try:
        if hasattr(trainer, 'attempt_num') and trainer.attempt_num:
            attempt_num = trainer.attempt_num
            attempt_num_path = os.path.join(training_args.output_dir, "attempt_num.json")
            with open(attempt_num_path, 'w') as f:
                json.dump(attempt_num, f, indent=4)
            logger.info(f"Attempt numbers saved to {attempt_num_path}")
        if hasattr(trainer, 'failed_items') and trainer.failed_items:
            failed_items = trainer.failed_items
            failed_items_path = os.path.join(training_args.output_dir, "failed_items.json")
            with open(failed_items_path, 'w') as f:
                json.dump(failed_items, f, indent=4)
            logger.info(f"Failed items saved to {failed_items_path}")
    except Exception as e:
        logger.error(f"Error saving failed items: {e}")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
