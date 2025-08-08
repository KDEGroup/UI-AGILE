import os
import json
import argparse
import logging
from collections import defaultdict
import math
from loguru import logger


import re
from typing import Optional, Any

def extract_param_value_loosely(model_response: str) -> Optional[str]:

    pattern = re.compile(
        r"""
        ['"](?P<key>\w+)['"]
        \s*:\s*
        (?P<value>
            \[.*?\] |        
            \{.*?\} |        
            ['"].*?['"] |    
            \w+              
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )
    value = None
    for match in pattern.finditer(model_response):
        key = match.group('key').lower() 
        value = match.group('value')

        if key != 'action':
            if (value.startswith("'") and value.endswith("'")) or \
               (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]
            
            return value

    return value

def gui_r1_extract_param(content: str) -> Optional[str]:
    pattern = r"'input_text':\s*'([^']*)'"
    match = re.search(pattern, content)
    if match:
        return match.group(1)
    return None


def calculate_f1_score(predicted_str, ground_truth_str):
    """Calculates the F1 score between two strings based on token overlap."""
    if not isinstance(predicted_str, str) or not isinstance(ground_truth_str, str):
        return 0.0
        
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if not predicted_tokens and not ground_truth_tokens:
        return 1.0
    if not predicted_tokens or not ground_truth_tokens:
        return 0.0

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    
    precision = len(common_tokens) / len(predicted_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

class Evaluator:
    """Encapsulates the logic for evaluating model predictions against ground truth."""

    # Define action categories based on the ground truth labels
    GROUNDING_ACTIONS = {'click', 'long_press', 'moveto', 'doubleclick', 'rightclick'}
    TEXT_ACTIONS = {'type', 'open_app', 'scroll', 'select'}
    SIMPLE_ACTIONS = {'press_back', 'wait', 'navigate_back', 'press_home', 'complete', 'impossible', 
                      'press_space', 'press_enter', 'press_down', 'hotkey', 'press_tab', 'press_pgdn'}

    def __init__(self, f1_threshold=0.5, grounding_tolerance=0.14):
        self.f1_threshold = f1_threshold
        self.grounding_tolerance = grounding_tolerance
        
        # Structure: { "category": { "metric": { "correct": 0, "total": 0 } } }
        # Added 'step_success' to track the new Success Rate metric.
        self.scores = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _is_grounding_correct(self, pred_coord, gt_bbox, image_size):
        """Checks if the predicted coordinate is within a normalized radius of the GT bbox center."""
        if not all(isinstance(v, (int, float)) for v in pred_coord + gt_bbox + image_size):
             return False
        if image_size[0] == 0 or image_size[1] == 0:
            return False
            
        pred_x, pred_y = pred_coord[:2]
        # logger.debug(f"pred_coord: {pred_coord}, gt_bbox: {gt_bbox}, image_size: {image_size}")
        gt_x, gt_y = gt_bbox[:2]
        
        # Normalized distance squared
        dist_sq = ((gt_x - pred_x) / image_size[0])**2 + ((gt_y - pred_y) / image_size[1])**2
        return dist_sq < self.grounding_tolerance**2

    def evaluate_prediction(self, pred_item):
        """
        Evaluates a single prediction item and updates scores.
        This function now calculates a new 'step_success' metric.
        """
        if not "gt_bbox" in pred_item:
            pred_item["gt_bbox"] = pred_item.get("gt_coordinate", None)
        if (pred_item["gt_bbox"] is None or pred_item["gt_bbox"] == []) and pred_item.get('gt_action') in self.GROUNDING_ACTIONS:
            logger.warning(f"Skipping item with missing gt_bbox: {pred_item}")
            return
        gt_action = pred_item.get('gt_action')
        pred_action = pred_item.get('pred_action')
        category = f"{pred_item.get('group')}-{gt_action}"

        # 1. Evaluate Action Type Accuracy (Unchanged)
        is_action_correct = (gt_action == pred_action)
        self.scores[category]['action']['total'] += 1
        if is_action_correct:
            self.scores[category]['action']['correct'] += 1

        # 2. Determine Overall Step Success (New Logic)
        self.scores[category]['step_success']['total'] += 1
        step_is_successful = False

        # For grounding actions, success requires correct action AND correct grounding.
        if gt_action in self.GROUNDING_ACTIONS:
            self.scores[category]['grounding']['total'] += 1
            is_grounding_correct = self._is_grounding_correct(
                pred_item['pred_coord'], 
                pred_item['gt_bbox'], 
                pred_item['image_size']
            )
            if is_grounding_correct:
                self.scores[category]['grounding']['correct'] += 1
            if is_action_correct and is_grounding_correct:
                step_is_successful = True

        # For text actions, success requires correct action AND correct text input.
        elif gt_action in self.TEXT_ACTIONS:
            self.scores[category]['text']['total'] += 1
            f1 = calculate_f1_score(pred_item.get('gt_input_text'), pred_item.get('pred_input_text'))
            is_text_correct = f1 >= self.f1_threshold
            if is_text_correct:
                self.scores[category]['text']['correct'] += 1
            if is_action_correct and is_text_correct:
                step_is_successful = True
            if is_action_correct and not is_text_correct:
                logger.debug(f"gt_input_text: {pred_item.get('gt_input_text')}, pred_param: {pred_item.get('pred_input_text')}, f1: {f1:.4f}")
                logger.debug(f"{pred_item.get('pred_raw')}")
                print("*" * 50)

        
        # For simple actions, success just requires the correct action.
        elif gt_action in self.SIMPLE_ACTIONS:
            if is_action_correct:
                step_is_successful = True

        if step_is_successful:
            self.scores[category]['step_success']['correct'] += 1
    
    def run(self, prediction_file_path):
        """Loads data and runs evaluation for all predictions."""
        with open(prediction_file_path, 'r') as f:
            for line in f:
                try:
                    self.evaluate_prediction(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")


    def report(self, logger):
        """Calculates and logs final scores in a formatted way."""
        
        overall_totals = defaultdict(lambda: defaultdict(int))
        for category, metrics in self.scores.items():
            for metric, counts in metrics.items():
                overall_totals[metric]['correct'] += counts.get('correct', 0)
                overall_totals[metric]['total'] += counts.get('total', 0)

        safe_div = lambda n, d: (n / d * 100) if d > 0 else 0.0

        # --- Detailed Report (Unchanged) ---
        logger.info("\n" + "="*25 + " Detailed Results by Category " + "="*25)
        for category in sorted(self.scores.keys()):
            action_acc = safe_div(self.scores[category]['action']['correct'], self.scores[category]['action']['total'])
            log_str = f"{category:<30} | Action Acc: {action_acc:6.2f}% ({self.scores[category]['action']['correct']}/{self.scores[category]['action']['total']})"
            
            if category.split('-')[1] in self.GROUNDING_ACTIONS:
                ground_acc = safe_div(self.scores[category]['grounding']['correct'], self.scores[category]['grounding']['total'])
                log_str += f" | Grounding Acc: {ground_acc:6.2f}% ({self.scores[category]['grounding']['correct']}/{self.scores[category]['grounding']['total']})"
            elif category.split('-')[1] in self.TEXT_ACTIONS:
                text_acc = safe_div(self.scores[category]['text']['correct'], self.scores[category]['text']['total'])
                log_str += f" | Text F1 Acc:   {text_acc:6.2f}% ({self.scores[category]['text']['correct']}/{self.scores[category]['text']['total']})"
            logger.info(log_str)

        # --- Overall Summary (Updated) ---
        logger.info("\n" + "="*28 + " Overall Summary " + "="*28)
        action_total = overall_totals['action']
        ground_total = overall_totals['grounding']
        text_total = overall_totals['text']
        success_total = overall_totals['step_success']

        logger.info(f"Overall Action Accuracy : {safe_div(action_total['correct'], action_total['total']):.2f}% ({action_total['correct']}/{action_total['total']})")
        logger.info(f"Overall Grounding Acc.  : {safe_div(ground_total['correct'], ground_total['total']):.2f}% ({ground_total['correct']}/{ground_total['total']})")
        logger.info(f"Overall Text F1 Acc.    : {safe_div(text_total['correct'], text_total['total']):.2f}% ({text_total['correct']}/{text_total['total']})")
        # Renamed "Step Accuracy" to "Success Rate" and using the new calculation.
        logger.info(f"Overall Success Rate    : {safe_div(success_total['correct'], success_total['total']):.2f}% ({success_total['correct']}/{success_total['total']})")
        logger.info("="*70)

# --- Main Execution (Unchanged) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, required=True, help="Path to the prediction file from main.py.")
    args = parser.parse_args()

    log_file = args.prediction_file_path.replace('.jsonl', '.log')

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Prevent duplicate logs if run multiple times
    logger.addHandler(logging.FileHandler(log_file, mode='a'))
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Prediction file: {os.path.basename(args.prediction_file_path)}")

    evaluator = Evaluator()
    evaluator.run(args.prediction_file_path)
    evaluator.report(logger)

if __name__ == '__main__':
    main()