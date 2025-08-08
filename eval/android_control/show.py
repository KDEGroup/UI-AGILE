from datasets import load_dataset
from datasets import Dataset
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from loguru import logger
from collections import defaultdict

"""
item structure:
keys: dict_keys(['image', 'history', 'instruction', 'gt_action', 'gt_bbox', 'gt_input_text', 'group', 'ui_type'])
values: [<class 'dict'>, <class 'str'>,  <class 'str'>, <class 'list'>, <class 'str'>, <class 'str'>, <class 'str'>]
"""
if __name__ == "__main__":
    data_path = 'GUI_Odyssey1000.parquet'
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    # logger.info(f"type of dataset: {type(dataset)}")
    # logger.info(f"keys of dataset: {dataset.keys()}")
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    unique_keys = set()
    keys_count = defaultdict(int)
    show_examples = []
    logger.info(f'keys: {dataset[0].keys()}')
    logger.info(f'values: {[type(value) for value in dataset[0].values()]}')
    
    for item in tqdm(dataset):
        # logger.info(f'keys: {item.keys()}')
        # value type
        # logger.info(f'values: {[type(value) for value in item.values()]}')
        item.pop("image", None)

        keys_count[item["gt_action"]] = keys_count[item["gt_action"]] + 1

        # if item["gt_action"] == "type":
        #     logger.debug(f"item: {item}")
        if not item["gt_action"] in unique_keys:
            unique_keys.add(item["gt_action"])
            show_examples.append(item)

    logger.info(f" actions found: {keys_count}")

    for example in show_examples:
        logger.info(f"instruction: {example['instruction']}")
        logger.info(f"history: {example.get('history', None)}")
        logger.info(f"gt_action: {example['gt_action']}")
        logger.info(f"gt_bbox: {example.get('gt_bbox', 'N/A')}")
        logger.info(f"gt_input_text: {example.get('gt_input_text', 'N/A')}")
        # logger.info(f"group: {example['group']}")
        # logger.info(f"ui_type: {example['ui_type']}")
        print("-" * 50)
