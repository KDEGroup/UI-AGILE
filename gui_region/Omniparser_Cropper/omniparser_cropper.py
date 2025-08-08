import torch
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
from tqdm import tqdm
from loguru import logger
import traceback
import time

# For DocTR
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from doctr.utils.geometry import detach_scores



def batch_predict_yolo(model: YOLO, images: list[Image.Image], box_threshold: float, iou_threshold=0.1) -> list[torch.Tensor]:
    """
    Runs YOLO prediction on a batch of images and returns normalized xyxy boxes.
    """
    results = model.predict(
        source=images,
        conf=box_threshold,
        iou=iou_threshold,
        verbose=False,  # Suppress verbose output
    )
    
    batch_boxes = []
    assert len(results) == len(images), "Number of results must match number of input images."
    for i, result in enumerate(results):
        w, h = images[i].size
        # Normalize boxes to [0, 1] range
        # logger.debug(f"shape of yolo boxes: {result.boxes.xyxy.shape}")
        if result.boxes.xyxy.numel() > 0:
            boxes = result.boxes.xyxy / torch.Tensor([w, h, w, h]).to(result.boxes.xyxy.device)
            # logger.debug(f"Normalized boxes for image {i}: {boxes}")
            batch_boxes.append(boxes)
        else:
            batch_boxes.append(torch.empty((0, 4)).to(result.boxes.xyxy.device))
        
    return batch_boxes


class BoundingBoxRefiner:
    """
    Refines a bounding box by finding the best match from a combined pool of
    YOLO (for objects/icons) and DocTR (for text) detections.
    """
    def __init__(self, yolo_model_path: str = "/ckpt/OmniParser-v2.0/icon_detect/model.pt", device: str = "cuda"):
        """
        Initializes the refiner by loading the YOLO and DocTR models.
        """
        _device = "cuda:1"
        # 1. Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(_device)

        # 2. Load DocTR detection model
        self.doc_predictor = detection_predictor(
            arch="db_resnet50",
            pretrained=True,
            assume_straight_pages=True,
            batch_size=16, # Adjust batch size based on your GPU memory
        ).to(_device)
        if 'cuda' in _device:
            self.doc_predictor.half()
        
        self.device = _device

    @staticmethod
    def iou(boxA: list[float], boxB: list[float]) -> float:
        """
        Calculates IoU for two boxes. Expects boxes in [x0, y0, x1, y1] format.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - interArea

        if union == 0:
            return 0.0
        return interArea / union

    def refine_bbox_batch(self, batch_inputs: list[list[str, Image.Image, tuple[int, int]]], threshold=0.02, extend_ratio=0.2) -> tuple[list[list[int] | None], list[float]]:
        """
        Processes a batch of images and center points to find refined bounding boxes.
        """
        if not batch_inputs:
            return []

        images = [item[1].convert('RGB') for item in batch_inputs]
        # --- Step 1: Batch YOLO Inference ---
        t1 = time.time()
        yolo_boxes_batch = batch_predict_yolo(self.yolo_model, images, box_threshold=0.05)
        # logger.debug(f"Batch YOLO inference completed in {time.time() - t1:.2f} seconds.")
        # --- Step 2: Batch DocTR Inference ---
        # doc_files should be numpy array
        t1 = time.time()
        doc_files = [np.array(img) for img in images]
        # logger.debug(f"Converting images to numpy arrays took {time.time() - t1:.2f} seconds.")
        t1 = time.time()
        doc_results = self.doc_predictor(doc_files)
        assert len(doc_results) == len(images), "DocTR results must match number of input images."
        # logger.debug(f"Batch DocTR inference completed in {time.time() - t1:.2f} seconds.")
        # --- Step 3: Combine and Normalize Detections ---
        # All boxes will be stored in normalized [0, 1] xyxy format.
        all_combined_boxes = []
        t1 = time.time()
        for i, res in enumerate(doc_results):
            # Add YOLO boxes (already normalized xyxy)
            combined_boxes_for_image = yolo_boxes_batch[i].tolist()
            # Add DocTR boxes
            detached_coords, _ = detach_scores([res.get("words")])
            for coords in detached_coords[0]:
                # logger.debug(f"DocTR coords : {coords}")
                coords = coords.reshape(2, 2).tolist() if coords.shape == (4,) else coords.tolist()
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                xyxy_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                # logger.debug(f"DocTR xyxy_box: {xyxy_box}")
                combined_boxes_for_image.append(xyxy_box)
            # clip to [0, 1] range
            combined_boxes_for_image = [[max(0, min(1, coord)) for coord in box] for box in combined_boxes_for_image]
            all_combined_boxes.append(combined_boxes_for_image)

        # --- Step 4: Final IoU Refinement ---
        final_results = []
        ious = []
        for i, (img_path, img, center_point) in enumerate(batch_inputs):
            try:
                img_width, img_height = img.size
                center_x, center_y = center_point

                # Create the preliminary bounding box in PIXEL coordinates
                box_width = img_width * 0.14
                box_height = img_height * 0.14
                px0 = max(0, int(center_x - box_width / 2))
                py0 = max(0, int(center_y - box_height / 2))
                px1 = min(img_width, int(center_x + box_width / 2))
                py1 = min(img_height, int(center_y + box_height / 2))
                pre_bbox_pixels = [px0, py0, px1, py1]

                # Convert pre_bbox to NORMALIZED coordinates for IoU calculation
                pre_bbox_norm = [px0 / img_width, py0 / img_height, px1 / img_width, py1 / img_height]
                
                combined_boxes = all_combined_boxes[i]
                if not combined_boxes:
                    final_results.append(pre_bbox_pixels) # Return original if no boxes found
                    continue

                # Find the box with the maximum IoU
                best_box_norm = pre_bbox_norm
                max_iou_score = threshold

                for norm_box in combined_boxes:
                    # logger.debug(f"Comparing pre_bbox {pre_bbox_norm} with norm_box {norm_box}")
                    current_iou = self.iou(pre_bbox_norm, norm_box)
                    if current_iou > max_iou_score:
                        max_iou_score = current_iou
                        best_box_norm = norm_box
                
                # extend the best box by extend_ratio
                norm_box_width = best_box_norm[2] - best_box_norm[0]
                norm_box_height = best_box_norm[3] - best_box_norm[1]
                extended_width = norm_box_width * (1 + extend_ratio)
                extended_height = norm_box_height * (1 + extend_ratio)
                best_box_norm[0] = max(0, best_box_norm[0] - (extended_width - norm_box_width) / 2)
                best_box_norm[1] = max(0, best_box_norm[1] - (extended_height - norm_box_height) / 2)
                best_box_norm[2] = min(1, best_box_norm[2] + (extended_width - norm_box_width) / 2)
                best_box_norm[3] = min(1, best_box_norm[3] + (extended_height - norm_box_height) / 2)

                # Convert the final NORMALIZED box back to PIXEL coordinates
                x0 = int(best_box_norm[0] * img_width)
                y0 = int(best_box_norm[1] * img_height)
                x1 = int(best_box_norm[2] * img_width)
                y1 = int(best_box_norm[3] * img_height)
                final_results.append([x0, y0, x1, y1])
                ious.append(max_iou_score)

            except Exception as e:
                logger.error(f"Error during post-processing refinement: {e}")
                logger.debug(traceback.format_exc())
                final_results.append(None) # Append None on error
        # logger.debug(f"Final refinement completed in {time.time() - t1:.2f} seconds.")
        return final_results, ious

    def get_element_img_batch(self, batch_inputs: list[list[str, Image.Image, tuple[int, int]]]) -> list[Image.Image]:
        """
        Processes a batch of images and center points to find refined element images.
        """
        batch_bboxs, _ = self.refine_bbox_batch(batch_inputs)
        refined_images = []
        for i, (img_path, img, center_point) in enumerate(batch_inputs):
            try:
                bbox = batch_bboxs[i]
                # Crop the image using the refined bounding box
                x0, y0, x1, y1 = bbox
                cropped_img = img.crop((x0, y0, x1, y1))
                refined_images.append(cropped_img)
            except Exception as e:
                logger.error(f"Error during image cropping: {e}")
                logger.debug(traceback.format_exc())
                refined_images.append(img)
        return refined_images


if __name__ == '__main__':
    import os
    import json
    # --- Configuration ---
    YOLO_MODEL_PATH = "OmniParser-v2.0/icon_detect/model.pt"
    IMAGE_DIR = "train_imgs"
    JSON_PATH = "train.json"
    
    # --- Prepare Input Data ---
    inputs = []
    gt_bboxes = []
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)
        json_data = json_data[:180]
    for item in json_data:
        bbox = item['bbox']
        gt_bboxes.append(bbox)
        point = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        inputs.append(['', Image.open(os.path.join(IMAGE_DIR, item['img_filename'])), point])

    refiner = BoundingBoxRefiner(
        yolo_model_path=YOLO_MODEL_PATH,
        device="cuda"
    )
        
    for threshold in [0, 0.01, 0.02, 0.05]:
        for extend_ratio in [0, 0.1, 0.2, 0.4]:
            print(f"Running refinement with threshold: {threshold}, extend_ratio: {extend_ratio}")
            refined_boxes, ious = refiner.refine_bbox_batch(inputs, threshold=threshold, extend_ratio=extend_ratio)
            print(f"mean iou: {np.array(ious).mean()}")
            
            # eval refined bboxes quality by comparing with ground truth through IoU
            iou_scores = []
            for i, (refined_box, gt_box) in enumerate(zip(refined_boxes, gt_bboxes)):
                if refined_box is None:
                    continue
                iou_score = BoundingBoxRefiner.iou(refined_box, gt_box)
                iou_scores.append(iou_score)
            print(f"refined bboxes quality iou score: {np.array(iou_scores).mean()}")
