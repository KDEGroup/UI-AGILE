import json
import os
from PIL import Image
# import math # math is imported but not explicitly used, can be removed if not needed by other logic
import copy # Use copy.deepcopy for nested structures if necessary
from loguru import logger

import torch
# Assuming logger is configured elsewhere, e.g.:
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def _generate_max_coords(original_dim: int, tile_dim: int, step_dim: int) -> list[int]:
    """
    Generates a sorted list of unique maximum coordinates for cropping.
    It includes coordinates from stepping and ensures the original dimension endpoint.
    """
    candidate_max_coords = []

    if original_dim >= tile_dim and step_dim > 0:
        candidate_max_coords.extend(list(range(tile_dim, original_dim, step_dim)))

    candidate_max_coords.append(original_dim)
    
    return sorted(list(set(candidate_max_coords)))

def process_image_cropping(
    original_pil_image: Image.Image,
    step_x: int,
    step_y: int,
    tile_width: int,
    tile_height: int
):
    # logger.info(f"sizeof original_pil_image: {original_pil_image.size}")
    output_tile_data = []
    original_width, original_height = original_pil_image.size

    all_crop_x_max = _generate_max_coords(original_width, tile_width, step_x)
    all_crop_y_max = _generate_max_coords(original_height, tile_height, step_y)

    for crop_x_max_val in all_crop_x_max:
        crop_x_min = max(crop_x_max_val - tile_width, 0)
        
        # Ensure the actual width of the crop is positive
        current_tile_width = crop_x_max_val - crop_x_min
        if current_tile_width <= 0:
            continue

        for crop_y_max_val in all_crop_y_max:
            crop_y_min = max(crop_y_max_val - tile_height, 0)

            # Ensure the actual height of the crop is positive
            current_tile_height = crop_y_max_val - crop_y_min
            if current_tile_height <= 0:
                continue

            try:
                actual_crop_box = (crop_x_min, crop_y_min, crop_x_max_val, crop_y_max_val)
                cropped_img = original_pil_image.crop(actual_crop_box)
                
                output_tile_data.append((cropped_img, (crop_x_min, crop_y_min)))
                
            except Exception as e:
                logger.error(f"Error processing tile data at crop box {actual_crop_box}: {e}")
                if 'cropped_img' in locals() and hasattr(cropped_img, 'close'):
                    cropped_img.close() # Ensure resource is released if an Image object was created
                continue # Continue to the next tile
                
    # logger.info(f"Total tiles processed: {len(output_tile_data)}")
    return output_tile_data


def get_element_sub_img(image, pred_coord, bbox_ratio=0.14, extra_area_ratio=0.2):
    """
    Args:
        image (PIL.Image): The input image to crop.
        pred_coord (list or tuple): The bounding box coordinates in the format [x1, y1, x2, y2] or point [x, y].

    Returns:
        list of PIL.Image: The cropped image.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if len(pred_coord) == 2:
        x, y = pred_coord
        bbox_w = image.width * bbox_ratio
        bbox_h = image.height * bbox_ratio
        x1 = max(0, x - bbox_w // 2)
        y1 = max(0, y - bbox_h // 2)
        x2 = min(image.width, x1 + bbox_w)
        y2 = min(image.height, y1 + bbox_h)
    elif len(pred_coord) == 4:
        x1, y1, x2, y2 = pred_coord
    new_width = (x2 - x1) * (1 + extra_area_ratio)
    new_height = (y2 - y1) * (1 + extra_area_ratio)
    new_x1 = max(0, x1 - (new_width - (x2 - x1)) // 2)
    new_y1 = max(0, y1 - (new_height - (y2 - y1)) // 2)
    new_x2 = min(image.width, new_x1 + new_width)
    new_y2 = min(image.height, new_y1 + new_height)
    pred_coord = [new_x1, new_y1, new_x2, new_y2]
    return image.crop(pred_coord)

