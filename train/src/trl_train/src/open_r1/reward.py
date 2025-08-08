import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_r1.utils import *
import re
import math
import traceback
from loguru import logger
import torch
import numpy as np
from typing import *


# 0-1
def action_reward(completions, solution,scales, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    instructions = kwargs["instruction"]
    for content, sol, instruction in zip(contents, solution, instructions):
        reward = 0.0

        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and ground_truth_action.lower() in student_answer_action.lower():
                reward = 1.0
        except Exception as e:
            logger.error(f"Error in action_reward: {e}")   
            traceback.print_exc()
        rewards.append(reward)

        logger.debug(f"instruction: {instruction} student_answer_action: {student_answer_action}, Ground Truth: {ground_truth_action}, Action Reward: {reward}")

    return rewards


def extract_param_value_loosely(model_response: str) -> Optional[str]:

    pattern = re.compile(
        r"""
        ['"](?P<key>\w+)['"]
        \s*:\s*
        (?P<value>
            \[.*?\] |         # [123, 456]
            \{.*?\} |         
            ['"].*?['"] |     # 'hello world'
            \w+               # down
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    for match in pattern.finditer(model_response):
        key = match.group('key').lower()  
        value = match.group('value')

        if key != 'action':
            if (value.startswith("'") and value.endswith("'")) or \
               (value.startswith('"') and value.endswith('"')):
                return value[1:-1]
            
            return value

    return None

def parameter_reward(completions, solution, scales, **kwargs):
    """Reward function that checks if the completion has the correct parameters."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # logger.debug(f"kwargs: {kwargs.keys()}")
    # kwargs: from dataset = Dataset.from_list(all_data) and dataset = dataset.map(make_conversation_from_json, num_proc=8)
    parameters = kwargs.get("parameter", None)
    if parameters is None or parameters == []:
        logger.warning("No parameters provided")
        return [0.0] * len(contents) 
    for content, sol, parameter in zip(contents, solution, parameters):
        reward = 0.0
        try:
            student_answer_param = extract_param_value_loosely(content)
            # logger.debug(f"Extracted student_answer_param: {student_answer_param}")
            if not parameter:
                reward = 0
            elif student_answer_param:
                if student_answer_param.lower() in parameter.lower():
                    reward = 1.0
            else:
                logger.warning(f"Failed to extract parameters from content: {content} or solution: {sol}")
        except Exception as e:
            logger.error(f"Error in parameter_reward: {e}")
            traceback.print_exc()
        rewards.append(reward)

    return rewards


# 0-1
def format_reward_uitars(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"Thought:.*?Action:.*?"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for match, content in zip(matches, completion_contents):
        if match:
            if len(extract_all_actions(content)) > 1:
                rewards.append(0)
                logger.warning(f"Multiple actions found in content: {content}")
            else:
                rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards




def check_thinking_format(content: str) -> bool:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if len(extract_all_actions(content)) > 1:
        logger.warning(f"Multiple actions found in content: {content}")
        return False
    return bool(pattern.fullmatch(content))

def check_nothinking_format(content: str) -> bool:
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    if len(extract_all_actions(content)) > 1:
        logger.warning(f"Multiple actions found in content: {content}")
        return False
    return bool(pattern.fullmatch(content))

def check_point_content(content: str) -> bool:
    coord, _, _ = extract_coordinates(content)
    return len(coord) == 2

def check_bbox_content(content: str) -> bool:
    coord, _, _ = extract_coordinates(content)
    return len(coord) == 4


FORMAT_CHECKERS: Dict[str, Callable[[str], bool]] = {
    "thinking": check_thinking_format,
    "simple_thinkingv2": check_thinking_format,
    "simple_thinking": check_thinking_format,
    "dast": check_thinking_format,
    "nothinking": check_nothinking_format,
}

CONTENT_CHECKERS: Dict[str, Callable[[str], bool]] = {
    "coord": check_point_content,
    "bbox": check_bbox_content,
}

# 0-1
def format_reward_uir1_wrapper(thinking_strategy: str, pred_type: str) -> Callable:

    try:
        format_checker = FORMAT_CHECKERS[thinking_strategy]
        content_checker = CONTENT_CHECKERS[pred_type]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported thinking strategy '{thinking_strategy}' or prediction type '{pred_type}'. "
        )
    def format_reward_uir1(completions: List[Dict], **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            content = comp[0]["content"]
            try:
                # 1. Perform the top-level format check (e.g., <think><answer> structure)
                if not format_checker(content):
                    rewards.append(0.0)
                    continue

                # 2. Extract action and convert to lowercase for case-insensitive comparison
                action_type = extract_action(content).lower()
                is_valid = False

                if not action_type:
                    rewards.append(0.0)
                    continue

                # 3. Perform action-specific validation
                if action_type == 'click':
                    # For 'click', rely entirely on the specific content checker (coord or bbox)
                    is_valid = content_checker(content)
                elif action_type in ['navigate_back', 'wait']:
                    # These actions are valid if the format is correct and action is identified
                    is_valid = True
                else:
                    # For actions with parameters, perform a loose check for their presence.
                    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                    if answer_match:
                        # Normalize the string inside <answer> to ignore case and quote style
                        answer_content = answer_match.group(1).lower().replace('"', "'")
                        
                        if action_type == 'scroll':
                            is_valid = "'direction':" in answer_content
                        elif action_type == 'input_text':
                            is_valid = "'input_text':" in answer_content
                        elif action_type == 'open_app':
                            is_valid = "'app':" in answer_content

                rewards.append(1.0 if is_valid else 0.0)

            except Exception as e:
                # In case of any unexpected error during processing, assign a reward of 0.
                # logger.error(f"Error processing content in format_reward_uir1: {e}")
                # logger.debug(f"Content that caused error: {content}")
                traceback.print_exc()
                rewards.append(0.0)
        return rewards

    return format_reward_uir1







import math

# 0-2
def calc_relative_coord_reward(x, y, bbox, decay_type="gaussian"):

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    if x < x1 or x > x2 or y < y1 or y > y2:
        return 0 
    
    width_half = (x2 - x1) / 2
    height_half = (y2 - y1) / 2
    
    norm_x_dist = abs(x - center_x) / width_half
    norm_y_dist = abs(y - center_y) / height_half
    

    norm_dist = max(norm_x_dist, norm_y_dist)
    
    if decay_type == "linear":
        reward = 1 - norm_dist
    
    elif decay_type == "quadratic":
        reward = 1 - norm_dist**2
    
    elif decay_type == "inverse_quadratic":
        reward = 1 - (norm_dist**2) / (1 + norm_dist**2)
    
    elif decay_type == "exponential":
        reward = math.exp(-3 * norm_dist)
    
    elif decay_type == "gaussian":
        reward = math.exp(-4 * norm_dist**2)
    
    elif decay_type == "cosine":
        reward = 0.5 * (1 + math.cos(math.pi * norm_dist))
    
    elif decay_type == "sigmoid":
        reward = 1 / (1 + math.exp(10 * (norm_dist - 0.5)))
    
    elif decay_type == "step":
        if norm_dist < 0.2:
            reward = 1.0
        elif norm_dist < 0.4:
            reward = 0.8
        elif norm_dist < 0.6:
            reward = 0.6
        elif norm_dist < 0.8:
            reward = 0.4
        else:
            reward = 0.2
    
    elif decay_type == "cubic":
        reward = 1 - norm_dist**3
    
    elif decay_type == "sqrt":
        reward = 1 - math.sqrt(norm_dist)
    
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    return 1 + max(0, min(1, reward))


# 0-2
def calc_bbox_reward(pred_bbox, target_bbox, method="iou", max_reward=2.0):

    import math
    
    x1 = max(pred_bbox[0], target_bbox[0])
    y1 = max(pred_bbox[1], target_bbox[1])
    x2 = min(pred_bbox[2], target_bbox[2])
    y2 = min(pred_bbox[3], target_bbox[3])
    
    if x2 < x1 or y2 < y1:
        intersection = 0
    else:
        intersection = (x2 - x1) * (y2 - y1)
    
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
    
    union = pred_area + target_area - intersection
    iou = intersection / union if union > 0 else 0
    
    if method.lower() == "iou":
        return iou * max_reward
    
    pred_center_x = (pred_bbox[0] + pred_bbox[2]) / 2
    pred_center_y = (pred_bbox[1] + pred_bbox[3]) / 2
    target_center_x = (target_bbox[0] + target_bbox[2]) / 2
    target_center_y = (target_bbox[1] + target_bbox[3]) / 2
    
    center_dist_squared = (pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2
    
    enclosing_x1 = min(pred_bbox[0], target_bbox[0])
    enclosing_y1 = min(pred_bbox[1], target_bbox[1])
    enclosing_x2 = max(pred_bbox[2], target_bbox[2])
    enclosing_y2 = max(pred_bbox[3], target_bbox[3])
    
    diagonal_dist_squared = (enclosing_x2 - enclosing_x1)**2 + (enclosing_y2 - enclosing_y1)**2
    
    if method.lower() == "diou":
        distance_penalty = center_dist_squared / diagonal_dist_squared if diagonal_dist_squared > 0 else 0
        diou = iou - distance_penalty
        

        return max(0, diou) * max_reward
    
    elif method.lower() == "ciou":
        pred_width = pred_bbox[2] - pred_bbox[0]
        pred_height = pred_bbox[3] - pred_bbox[1]
        target_width = target_bbox[2] - target_bbox[0]
        target_height = target_bbox[3] - target_bbox[1]
        
        if pred_height <= 0 or target_height <= 0:
            aspect_term = 0
        else:
            pred_arctan = math.atan(pred_width / pred_height)
            target_arctan = math.atan(target_width / target_height)
            aspect_term = 4 / (math.pi**2) * ((target_arctan - pred_arctan)**2)
        
        v = aspect_term
        alpha = v / (1 - iou + v) if 0.01 < iou < 0.99 else 0
        
        distance_penalty = center_dist_squared / diagonal_dist_squared if diagonal_dist_squared > 0 else 0
        ciou = iou - distance_penalty - alpha * v
        
        return max(0, ciou) * max_reward
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'iou', 'diou', and 'ciou'.")











def grounding_reward(completions, solution, scales, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol, scale in zip(contents, solution, scales):
        reward = 0.0
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and ground_truth_action.lower() in student_answer_action.lower():
                if ground_truth_action == "click":
                    ground_truth_bbox, flag2= extract_bbox(sol)
                    student_answer_coord, flag1, _ = extract_coordinates(content)
                    if len(ground_truth_bbox) == 2:
                        # bbox_w = 100
                        # bbox_h = 60
                        # ground_truth_bbox = [ground_truth_bbox[0] - bbox_w / 2, ground_truth_bbox[1] - bbox_h / 2,
                        #                      ground_truth_bbox[0] + bbox_w / 2, ground_truth_bbox[1] + bbox_h / 2]
                        raise ValueError(f"ground_truth_bbox should be a bbox, but got {ground_truth_bbox}")
                    if len(student_answer_coord) == 2:
                        student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1])]
                        reward = calc_relative_coord_reward(student_answer_coord[0], student_answer_coord[1], ground_truth_bbox)
                    elif len(student_answer_coord) == 4:
                        student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1]),
                                                int(student_answer_coord[2] * scale[0]), int(student_answer_coord[3] * scale[1])]
                        reward = calc_bbox_reward(student_answer_coord, ground_truth_bbox)
                    # logger.debug(f"student_answer_coord: {student_answer_coord}, ground_truth_bbox: {ground_truth_bbox}")
                else: 
                    reward = 1.0
            elif ground_truth_action != "click":
                reward = 0.0
                logger.warning(f"action not match, student: {student_answer_action}, ground truth: {ground_truth_action}")                
            else:   
                reward = 0.0
                logger.warning(f"action not match, student: {student_answer_action}, ground truth: {ground_truth_action}")
        except Exception:
            pass  # Continue to next verification method if this fails
        # logger.debug(f"reward: {reward}")

        rewards.append(reward)

    return rewards



def binary_grounding_reward(completions, solution, scales, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol, scale in zip(contents, solution, scales):
        reward = 0.0
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and ground_truth_action.lower() in student_answer_action.lower():
                if ground_truth_action == "click":
                    ground_truth_bbox, flag2= extract_bbox(sol)
                    student_answer_coord, flag1, _ = extract_coordinates(content)
                    if len(ground_truth_bbox) == 2:
                        raise ValueError(f"ground_truth_bbox should be a bbox, but got {ground_truth_bbox}")
                    if len(student_answer_coord) == 2:
                        student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1])]
                        reward = student_answer_coord[0] >= ground_truth_bbox[0] and student_answer_coord[0] <= ground_truth_bbox[2] and \
                                student_answer_coord[1] >= ground_truth_bbox[1] and student_answer_coord[1] <= ground_truth_bbox[3] 
                        reward = 1 if reward else 0
                    elif len(student_answer_coord) == 4:
                        student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1]),
                                                int(student_answer_coord[2] * scale[0]), int(student_answer_coord[3] * scale[1])]
                        reward = student_answer_coord[0] >= ground_truth_bbox[0] and student_answer_coord[1] >= ground_truth_bbox[1] and \
                                student_answer_coord[2] <= ground_truth_bbox[2] and student_answer_coord[3] <= ground_truth_bbox[3]
                        reward = 1 if reward else 0   
                    # logger.debug(f"student_answer_coord: {student_answer_coord}, ground_truth_bbox: {ground_truth_bbox}")
                else: 
                    reward = 1.0
            elif ground_truth_action != "click": 
                reward = 0.0
                logger.warning(f"action not match, student: {student_answer_action}, ground truth: {ground_truth_action}")                
            else:
                reward = 0.0
                logger.warning(f"action not match, student: {student_answer_action}, ground truth: {ground_truth_action}")
        except Exception:
            pass  # Continue to next verification method if this fails
        # logger.debug(f"reward: {reward}")

        rewards.append(reward)

    return rewards






def simple_length_reward(completions, solution, scales, **kwargs):


    g_rewards = grounding_reward(completions, solution, scales, **kwargs)
    contents = [completion[0]["content"] for completion in completions]
    
    l_ideal = kwargs.get("l_ideal", 150)
    sigma = kwargs.get("sigma", l_ideal / 2)

    final_rewards = []
    for content, g_reward in zip(contents, g_rewards):
        reward = 0.0
        if g_reward > 0:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                think_part = think_match.group(1).strip()
                L = len(think_part)
                diff = L - l_ideal
                reward = math.exp(- (diff ** 2) / (2 * sigma ** 2))
            else:
                reward = 0.0
        
        final_rewards.append(reward)
        
    return final_rewards


def plain_length_reward(completions, solution, scales, **kwargs):


    g_rewards = grounding_reward(completions, solution, scales, **kwargs)
    contents = [completion[0]["content"] for completion in completions]

    l_min = kwargs.get("l_min", 50)
    l_ideal_start = kwargs.get("l_ideal_start", 120)
    l_ideal_end = kwargs.get("l_ideal_end", 200)
    l_max = kwargs.get("l_max", 300)
    VALID_ENDINGS = ('.', '。', '?', '？', '!', '！', '…')
    COMPLETENESS_BONUS = kwargs.get("completeness_bonus", 0.2)
    LENGTH_REWARD_MAX = kwargs.get("length_reward_max", 0.8) 

    def smooth_transition(x):
        return (1 - math.cos(x * math.pi)) / 2

    final_rewards = []
    for content, g_reward in zip(contents, g_rewards):
        reward = 0.0
        
        if g_reward > 0:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            
            if think_match:
                think_part = think_match.group(1).strip()
                L = len(think_part)
                
                length_reward = 0.0
                if l_ideal_start < L <= l_ideal_end:
                    length_reward = 1.0
                elif l_min < L <= l_ideal_start:
                    x_up = (L - l_min) / (l_ideal_start - l_min)
                    length_reward = smooth_transition(x_up)
                elif l_ideal_end < L < l_max:
                    x_down = (L - l_ideal_end) / (l_max - l_ideal_end)
                    length_reward = 1.0 - smooth_transition(x_down)
                
                completeness_bonus = 0.0
                if think_part.endswith(VALID_ENDINGS):
                    completeness_bonus = COMPLETENESS_BONUS
                
                reward = length_reward * LENGTH_REWARD_MAX + completeness_bonus
        
        final_rewards.append(reward)
        
    return final_rewards


  


def length_reward_dast(rewards: torch.Tensor, completion_token_length: torch.Tensor, num_generations, dast_a=-0.5, dast_b=0.5) -> torch.Tensor:
    assert rewards.size(0) == completion_token_length.size(0), "rewards and completion_token_length must have the same size"

    # reshape  (batch_size, num_generations)
    B, N = -1, num_generations
    rewards = rewards.view(B, N).float()
    lengths = completion_token_length.view(B, N).float()

    correct = rewards > 2

    # shape: (batch_size,)
    p = correct.sum(dim=1) / N  # (B,)

    # Lr: shape: (B,)
    Lr = torch.where(
        correct.any(dim=1),
        (lengths * correct).sum(dim=1) / correct.sum(dim=1).clamp(min=1),
        torch.zeros_like(p)
    )

    # Lmax: shape: (B,)
    Lmax = lengths.max(dim=1).values

    # Lb: p * Lr + (1 - p) * Lmax
    Lb = p * Lr + (1 - p) * Lmax  # (B,)

    # lambdas = (L - Lb) / Lb
    lambdas = (lengths - Lb.unsqueeze(1)) / Lb.unsqueeze(1)  # (B, N)

    # add rewards based on reward>2 mask
    add_rewards = torch.where(
        correct,
        torch.clamp(dast_a * lambdas + dast_b, min=0.1),
        torch.clamp(0.9 * lambdas - 0.1, max=-0.1)
    )

    return add_rewards.view(-1)



if __name__ == "__main__":
    # test format_reward_uir1_wrapper
    completions = [[{"content": "<think>Do something</think><answer>[{'action': 'input_text', 'input_text': 'up'}]</answer>"}]]
    kwargs = {
        "thinking_strategy": "thinking",
        "pred_type": "coord"
    }
    reward_func = format_reward_uir1_wrapper(kwargs["thinking_strategy"], kwargs["pred_type"])
    rewards = reward_func(completions, **kwargs)
    print(rewards) 