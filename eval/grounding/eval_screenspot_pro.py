import copy
import itertools

import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import traceback
logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)
from loguru import logger
GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--pred_type', type=str, default="nothink_point", choices=["think_point", "nothink_point", "think_bbox", "nothink_bbox"], help="Type of prediction: point or bbox.")
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--vllm_api', action="store_true", help="Whether to use vllm API.")
    parser.add_argument('--generate_score_eval_data', action="store_true", help="Whether to generate score eval data.")
    parser.add_argument('--port', type=int, default=8000, help="Port for vllm API.")
    parser.add_argument('--vllm_model_name', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model name for vllm API.")
    parser.add_argument('--vllm_local', action="store_true", help="Whether to use vllm local.")
    parser.add_argument('--score_method', default="guiclip", choices=["guiclip", "colqwen", "uiclip", "dual_encoder"], help="Method to calculate scores for cropping.")
    parser.add_argument('--step_ratio', type=float, default=0.5, help="Step ratio for cropping.")
    parser.add_argument('--tile_ratio', type=float, default=0.6, help="Tile size for cropping.")
    args = parser.parse_args()
    assert not (args.vllm_api and args.vllm_local), "You can only use one of vllm_api or vllm_local."

    return args

def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "cogagent":
        from models.cogagent import CogAgentModel
        model = CogAgentModel()
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel
        model = SeeClickModel()
    elif model_type == "qwen1vl":
        from models.qwen1vl import Qwen1VLModel
        model = Qwen1VLModel()
    elif model_type == "uitars":
        from models.uitars import UITarsModel
        model = UITarsModel()
    elif model_type == "uitars1.5":
        from models.uitars1_5 import UITars1_5Model
        model = UITars1_5Model()
    elif model_type == "uitars1.5_crop_select":
        from models.uitars1_5_crop_select import UITars1_5_CropSelect
        model = UITars1_5_CropSelect()
    elif model_type == "uir1":
        from models.uir1 import UI_R1
        model = UI_R1()
    elif model_type == "uir1_crop_select":
        from models.uir1_crop_select import UI_R1_CropSelect
        model = UI_R1_CropSelect()
    elif model_type == "guir1":
        from models.guir1 import GUI_R1
        model = GUI_R1()
    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel
        model = MiniCPMVModel()
    elif model_type == "internvl":
        from models.internvl import InternVLModel
        model = InternVLModel()
    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import GPT4XModel
        model = GPT4XModel()
    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel
        model = OSAtlas4BModel()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel
        model = OSAtlas7BModel()
    elif model_type == "uground":
        from models.uground import UGroundModel
        model = UGroundModel()
    elif model_type == "fuyu":
        from models.fuyu import FuyuModel
        model = FuyuModel()
    elif model_type == "showui":
        from models.showui import ShowUIModel
        model = ShowUIModel()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIVLLMModel
        model = AriaUIVLLMModel()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model
        model = CogAgent24Model()
    elif model_type == "seeclick-pro-agent":
        from models.seeclick_pro import SeeClickProAgent
        from models.osatlas7b import OSAtlas7BVLLMModel
        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = SeeClickProAgent(grounder=grounder)
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.load_model(model_name_or_path=model_name_or_path, args=args)
    return model

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results


def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations


def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2

    click_point = response.get("point", None)  # may be none
    pred_bbox = response.get("bbox", None)
    if click_point is not None:
        if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
            return "correct"
    elif pred_bbox is not None:
        point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
        if (bbox[0] <= point[0] <= bbox[2]) and (bbox[1] <= point[1] <= bbox[3]):
            return "correct"
    else:
        return "wrong"
    
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    try:
        result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    except Exception as e:
        logger.error(f"Error in evaluating fine-grained metrics: {e}")
        traceback.print_exc()
    try:
        result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    except Exception as e:
        logger.error(f"Error in evaluating SeeClick paper style metrics: {e}")
        traceback.print_exc()
    try:
        result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    except Exception as e:
        logger.error(f"Error in evaluating leaderboard simple style metrics: {e}")
        traceback.print_exc()
    try:
        result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    except Exception as e:
        logger.error(f"Error in evaluating leaderboard detailed style metrics: {e}")
        traceback.print_exc()
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report

def main(args):
    model = build_model(args)
    print("Load model success")

    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)

        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                # TODO: Translate the data
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    print(f"Total tasks: {len(tasks_to_run)}")

    results = []

    if args.vllm_local:
        responses = model.ground_only_positive_batch(instructions=[sample["prompt_to_evaluate"] for sample in tasks_to_run], 
                                                     images=[os.path.join(args.screenspot_imgs, sample["img_filename"]) for sample in tasks_to_run])

    log_step = 5
    score_evals = []
    for i in tqdm(range(len(tasks_to_run))):
        sample = tasks_to_run[i]
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if args.vllm_local:
            response = responses[i]
        else:
            try:
                if task_instance["gt_type"] == "positive":
                    if args.vllm_api:
                        if args.generate_score_eval_data:
                            score_eval_base_dir = f"./score_eval_{args.vllm_model_name}_{args.score_method}_{args.pred_type}"
                            response, score_eval = model.generate_score_eval_data(
                                instruction=sample["prompt_to_evaluate"], 
                                image=img_path, 
                                sample=sample,
                                args=args,
                                image_save_dir=os.path.join(score_eval_base_dir, "images")
                            )
                            if score_eval is not None:
                                score_evals.append(score_eval)
                            # logger.debug(f"score_eval: {score_eval} response: {response}")
                        else:
                            response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path, args=args)
                    else:
                        response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path, vllm_api=args.vllm_api)
                elif task_instance["gt_type"] == "negative":
                    response = model.ground_allow_negative(instruction=sample["prompt_to_evaluate"], image=img_path)
            except Exception as e:
                traceback.print_exc()
        point = response["point"]
        point_in_pixel = point
        sample_result = {
            "img_path": img_path, 
            "group": sample["group"] if "group" in sample else None,
            # "platform": sample["platform"],
            "platform": sample["platform"] if "platform" in sample else sample["data_source"],
            "application": sample["application"] if "application" in sample else None,
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"], 
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"] if "ui_type" in sample else sample["data_type"], 
            "task_filename": sample["task_filename"], 
            "pred": point_in_pixel, 
            "raw_response": response["raw_response"]
        }
        logger.info(f"bbox:{sample['bbox']}, point_in_pixel: {point_in_pixel}")
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result.update({
                "bbox": sample["bbox"], 
            })
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong instruction type")

        
        sample_result.update({
            "correctness": correctness,
        })
        results.append(sample_result)
        if (i + 1) % log_step == 0:
            logging.info(f"Processed {i + 1} samples.")
            logging.info(f"{evaluate_overall(results)}")
            logger.info(f"n pass acc: {len(score_evals)/ (i + 1)}")
    result_report = evaluate(results)
    # Save to file
    try:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, 'w') as f:
            json.dump(result_report, f, indent=4)
        logging.info("Evaluation of ScreenSpot finished.")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")
        logging.error(traceback.format_exc())

    with open(os.path.join(score_eval_base_dir, "score_eval.json"), 'w') as f:
        json.dump(score_evals, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
