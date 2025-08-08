import json
import os
from PIL import Image
# import math # math is imported but not explicitly used, can be removed if not needed by other logic
import copy # Use copy.deepcopy for nested structures if necessary
from loguru import logger

TILE_WIDTH = 2560  
TILE_HEIGHT = 1440 

RESIZE_WIDTH = 2560
RESIZE_HEIGHT = 1440
# High-quality resampling filter for resizing
# Pillow versions >= 9.1.0 use Image.Resampling.LANCZOS
# Older versions use Image.LANCZOS
RESAMPLING_FILTER = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS


ANNOTATIONS_DIR = './'
IMAGES_DIR = 'train_imgs'      
OUTPUT_ANNOTATIONS_DIR = f'clipped/new/{TILE_WIDTH}'
OUTPUT_IMAGES_DIR = f'clipped/new/{TILE_WIDTH}/train_imgs'       


def _get_output_paths(base_filename: str, ext: str, img_subdir: str, output_base_dir: str, suffix: str = ""):

    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix
    output_filename = f"{base_filename}{suffix}{ext}"
    output_rel_path = os.path.join(img_subdir, output_filename)
    output_full_path = os.path.join(output_base_dir, output_rel_path)
    output_dir = os.path.dirname(output_full_path)
    return output_full_path, output_rel_path, output_dir



def process_image_resize( 
    original_pil_image: Image.Image,
    target_width: int = RESIZE_WIDTH,
    target_height: int = RESIZE_HEIGHT
):
    """

    Args:
        original_pil_image (Image.Image)
        target_width (int)
        target_height (int)

    Returns:
        tuple: (resized_pil_image, new_width, new_height, operation_suffix_type) or (None, None, None, None)。
               `operation_suffix_type` is a generic string like "resized" for the caller.
    """
    if original_pil_image.width < target_width or original_pil_image.height < target_height:
        logger.warning(f"Original size ({original_pil_image.width}x{original_pil_image.height}) smaller than taget size ({target_width}x{target_height})")
        return None, None, None, None
    try:
        resized_img = original_pil_image.resize((target_width, target_height), resample=RESAMPLING_FILTER)
        operation_suffix_type = "resized"
        return resized_img, target_width, target_height, operation_suffix_type
    except Exception as e:
        logger.error(f"{e}")
        return None, None, None, None


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



def process_image_cropping(
    original_pil_image: Image.Image,
    original_bbox: list,
    tile_width = TILE_WIDTH,
    tile_height = TILE_HEIGHT
):
    """
    Returns:
        list:
            - 'cropped_pil_image': PIL Image
            - 'new_bbox'
            - 'actual_tile_width'
            - 'actual_tile_height'
            - 'crop_coords':  [crop_x_min, crop_y_min, crop_x_max, crop_y_max]。
            - 'coordinate_suffix'
    """
    if not original_bbox or len(original_bbox) != 4:
        logger.warning(f"Invalid bbox: {original_bbox}")
        return None
    if original_pil_image.width < tile_width or original_pil_image.height < tile_height:
        logger.warning(
            f"Original size ({original_pil_image.width}x{original_pil_image.height}) smaller than "
            f"({tile_width}x{tile_height})"
        )
        return None
    orig_x_min, orig_y_min, orig_x_max, orig_y_max = original_bbox
    processed_crops_info = []

    original_width, original_height = original_pil_image.size

    bbox_width = orig_x_max - orig_x_min
    bbox_height = orig_y_max - orig_y_min

    if bbox_width <= 0 or bbox_height <= 0:
        logger.warning(f"Invalid bbox size: w={bbox_width}, h={bbox_height}. Bbox: {original_bbox}")
        return None


    if bbox_width > tile_width or bbox_height > tile_height:
        logger.warning(
            f"Bbox ({bbox_width}x{bbox_height}) is larger than tile size "
            f"({tile_width}x{tile_height}). Cropping might not fully contain the bbox as intended "
            f"with current tiling logic for bbox: {original_bbox}"
        )
        return None
    
    step_x = int(tile_width - bbox_width)
    step_y = int(tile_height - bbox_height)
    original_width = int(original_width)
    original_height = int(original_height)
    tile_width = int(tile_width)
    tile_height = int(tile_height)
    all_crop_x_max = _generate_max_coords(original_width, tile_width, step_x)
    all_crop_y_max = _generate_max_coords(original_height, tile_height, step_y)

    for crop_x_max in all_crop_x_max:
        crop_x_min = max(crop_x_max - tile_width, 0)
        for crop_y_max in all_crop_y_max:
            crop_y_min = max(crop_y_max - tile_height, 0)

    # for crop_x_min in range(0, original_width, step_x):
    #     crop_x_max = min(crop_x_min + tile_width, original_width)
    #     for crop_y_min in range(0, original_height, step_y):
    #         crop_y_max = min(crop_y_min + tile_height, original_height)

            is_contained = (orig_x_min >= crop_x_min and
                            orig_y_min >= crop_y_min and
                            orig_x_max <= crop_x_max and
                            orig_y_max <= crop_y_max)

            if is_contained:
                try:
                    actual_crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
                    cropped_img = original_pil_image.crop(actual_crop_box)

                    actual_tile_width, actual_tile_height = cropped_img.size

                    if actual_tile_width == 0 or actual_tile_height == 0:
                        logger.warning(f"Empty tile {actual_crop_box} for bbox {original_bbox}")
                        cropped_img.close()
                        continue
                    
                    coordinate_suffix = f"x{crop_x_min}_y{crop_y_min}"

                    new_bbox_x_min = orig_x_min - crop_x_min
                    new_bbox_y_min = orig_y_min - crop_y_min
                    new_bbox_x_max = orig_x_max - crop_x_min
                    new_bbox_y_max = orig_y_max - crop_y_min
                    
                    new_bbox = [new_bbox_x_min, new_bbox_y_min, new_bbox_x_max, new_bbox_y_max]

                    # Handle resizing if the cropped tile is too small
                    if actual_tile_width <= 28 or actual_tile_height <= 28:
                        # Store original (pre-resize) dimensions
                        pre_resize_width, pre_resize_height = actual_tile_width, actual_tile_height

                        # Resize the image
                        cropped_img = resize_image(cropped_img, min_size=28)
                        
                        # Get new (post-resize) dimensions
                        post_resize_width, post_resize_height = cropped_img.size
                        
                        # Calculate scaling factors
                        scale_w = post_resize_width / pre_resize_width
                        scale_h = post_resize_height / pre_resize_height
                        
                        # Recalculate new_bbox by applying scaling factors
                        new_bbox_x_min = int(new_bbox[0] * scale_w)
                        new_bbox_y_min = int(new_bbox[1] * scale_h)
                        new_bbox_x_max = int(new_bbox[2] * scale_w)
                        new_bbox_y_max = int(new_bbox[3] * scale_h)

                        new_bbox = [new_bbox_x_min, new_bbox_y_min, new_bbox_x_max, new_bbox_y_max]

                        # The 'actual_tile_width/height' should now reflect the resized dimensions
                        actual_tile_width, actual_tile_height = post_resize_width, post_resize_height


                    processed_crops_info.append({
                        'cropped_pil_image': cropped_img,
                        'new_bbox': [new_bbox_x_min, new_bbox_y_min, new_bbox_x_max, new_bbox_y_max],
                        'actual_tile_width': actual_tile_width,
                        'actual_tile_height': actual_tile_height,
                        'crop_coords': [crop_x_min, crop_y_min, crop_x_max, crop_y_max],
                        'coordinate_suffix': coordinate_suffix
                    })
                except Exception as e:
                    logger.error(f"Error ({crop_x_min},{crop_y_min}) for bbox {original_bbox} : {e}")
                    if 'cropped_img' in locals() and hasattr(cropped_img, 'close'):
                        cropped_img.close()
                    continue
    return processed_crops_info



def process_dataset_cropd_filtered(annotation_filename: str):

    input_json_path = os.path.join(ANNOTATIONS_DIR, annotation_filename)
    output_json_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, annotation_filename)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found Error {input_json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Parsing json Error {input_json_path}")
        return

    all_new_annotations = []

    for item_index, item in enumerate(original_data):
        img_rel_path = item.get('img_filename')
        original_bbox = item.get('bbox')
        item_id = item.get('id', f'item_{item_index}')

        if not img_rel_path:
            logger.warning(f"Lack of 'img_filename'")
            continue

        original_img_full_path = os.path.join(IMAGES_DIR, img_rel_path)
        base_filename, ext = os.path.splitext(os.path.basename(img_rel_path))
        img_subdir = os.path.dirname(img_rel_path)

        original_pil_image = None
        try:
            original_pil_image = Image.open(original_img_full_path)

            is_valid_bbox = original_bbox and len(original_bbox) == 4 and \
                            (original_bbox[2] > original_bbox[0]) and \
                            (original_bbox[3] > original_bbox[1])

            if not is_valid_bbox:
                logger.info(f" Skip : {item_id} ( {img_rel_path}) - invalid bbox: {original_bbox})")
                resized_pil_img, new_width, new_height, op_suffix_type = process_image_resize(
                    original_pil_image, RESIZE_WIDTH, RESIZE_HEIGHT
                )
                if resized_pil_img:
                    # Construct item-specific suffix using item_id and op_suffix_type
                    item_specific_resize_suffix = f"{op_suffix_type}_{item_id}"
                    
                    output_full_path, output_rel_path, output_dir = _get_output_paths(
                        base_filename, ext, img_subdir, OUTPUT_IMAGES_DIR, suffix=item_specific_resize_suffix
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    resized_pil_img.save(output_full_path)
                    resized_pil_img.close()

                    new_item = copy.deepcopy(item)
                    new_item['img_filename'] = output_rel_path
                    new_item['img_size'] = [new_width, new_height]
                    new_item['processing_method'] = op_suffix_type # e.g., 'resized'
                    new_item.pop('bbox', None)
                    all_new_annotations.append(new_item)
                else:
                    logger.warning(f"Can not process ID: {item_id}. Skipped")
            else:
                logger.info(f" Processing ID: {item_id} ({img_rel_path}) - with bbox {original_bbox}")
                processed_crops_data_list = process_image_cropping(
                    original_pil_image, original_bbox, TILE_WIDTH, TILE_HEIGHT
                )
                crops_saved_for_item = 0
                if processed_crops_data_list:
                    for crop_data in processed_crops_data_list:
                        cropped_pil_image = crop_data['cropped_pil_image']
                        item_specific_crop_suffix = f"crop_{item_id}_{crop_data['coordinate_suffix']}"
                        
                        output_full_path, output_rel_path, output_dir = _get_output_paths(
                            base_filename, ext, img_subdir, OUTPUT_IMAGES_DIR, suffix=item_specific_crop_suffix
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        cropped_pil_image.save(output_full_path)
                        cropped_pil_image.close()

                        new_item = copy.deepcopy(item)
                        new_item['img_filename'] = output_rel_path
                        new_item['bbox'] = crop_data['new_bbox']
                        new_item['img_size'] = [crop_data['actual_tile_width'], crop_data['actual_tile_height']]
                        new_item['processing_method'] = "cropped" # Hardcoded as 'cropped' for this path
                        new_item['original_img_filename'] = img_rel_path
                        new_item['crop_coords'] = crop_data['crop_coords']
                        all_new_annotations.append(new_item)
                        crops_saved_for_item += 1
                logger.info(f"Saved {crops_saved_for_item} for ID {item_id} with (bbox: {original_bbox})")

        except FileNotFoundError:
            logger.error(f"Image not found {original_img_full_path} for item ID {item_id}。")
        except Exception as e:
            logger.error(f"Error: ID {item_id}  {original_img_full_path} {e}", exc_info=True)
        finally:
            if original_pil_image:
                original_pil_image.close()

    if all_new_annotations:
        try:
            all_new_annotations.sort(key=lambda x: (
                x.get('id', ''),
                x.get('crop_coords', [0,0,0,0])[1] if x.get('processing_method') == 'cropped' else 0,
                x.get('crop_coords', [0,0,0,0])[0] if x.get('processing_method') == 'cropped' else 0
            ))
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_new_annotations, f, indent=4, ensure_ascii=False)
            logger.info(f"Completed {annotation_filename}, generated {len(all_new_annotations)} new anno, saved to {output_json_path}")
        except Exception as e:
            logger.error(f"Error writing {output_json_path}: {e}", exc_info=True)
    else:
        logger.info(f"{annotation_filename} no valid anno")

if __name__ == "__main__":
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

    if not os.path.isdir(ANNOTATIONS_DIR):
        logger.error(f"'{ANNOTATIONS_DIR}' not found")
    elif not os.path.isdir(IMAGES_DIR):
        logger.error(f"'{IMAGES_DIR}' not found")
    else:
        json_files_found = False
        for filename in os.listdir(ANNOTATIONS_DIR):
            if filename.endswith(".json"):
                json_files_found = True
                process_dataset_cropd_filtered(filename)
        if not json_files_found:
            logger.warning(f" .json not found in '{ANNOTATIONS_DIR}'")
        logger.info("Completed")