import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Union, Tuple, Optional, List

from utils.util import morphology_clean, preprocess_image_segment

import cv2
import logging

logger = logging.getLogger(__name__)


def expand_box(x1, y1, x2, y2, pad_ratio, shape):
    H, W = shape[:2]
    w, h = x2 - x1, y2 - y1
    dx, dy = w * pad_ratio, h * pad_ratio
    nx1 = max(0, int(np.floor(x1 - dx)))
    ny1 = max(0, int(np.floor(y1 - dy)))
    nx2 = min(W, int(np.ceil(x2 + dx)))
    ny2 = min(H, int(np.ceil(y2 + dy)))
    return nx1, ny1, nx2, ny2

class SAM2Segmenter:
  """
    SAM2 (Segment Anything Model 2) for precise food segmentation.
    Takes bounding boxes from detection and generates accurate masks.
  """

  def __init__(self, cfg: Dict):
    self.cfg = cfg
    self.model = None
    self.predictor = None
    self.image_embedding = None

    # SAM2 config
    sam_cfg = cfg.get('sam2', {})
    self.model_type = sam_cfg.get('model_type', 'sam2_hiera_large')
    self.checkpoint_path = sam_cfg.get('checkpoint_path', None)
    self.device = sam_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    self.enable_morph = sam_cfg.get("morphology", True)
    self.enable_preproc = sam_cfg.get("preprocess", True)

    logger.info(f"SAM2Segmenter initialized with {self.model_type}")



  def load_model(self, checkpoint_path: Optional[Union[str, Path]] = None):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = checkpoint_path or self.checkpoint_path
    if ckpt is None:
      raise ValueError("SAM checkpoint path is not provided")

    ckpt = Path(ckpt)
    if not ckpt.exists():
      logger.warning(f"SAM2 checkpoint not found: {ckpt}")
      return
      
    self.model = build_sam2(
      config_file=self.model_type,
      ckpt_path=str(ckpt),
      device=self.device,
    )
    self.predictor = SAM2ImagePredictor(self.model)


  def set_image(self, image: np.ndarray):
    if self.enable_preproc:
      image = preprocess_image_segment(image)
    self.predictor.set_image(image)
    self.image_embedding = self.predictor.get_image_embedding()

  def segment_bbox(
    self,
    bbox: Union[Dict, Tuple[float, float, float, float]],
    use_center_prompt=True,
    multimask_output=False
  ) -> Dict:
    if self.predictor is None or self.image_embedding is None:
      raise RuntimeError("Image embedding is not set, call set_image() first")
    
    if isinstance(bbox, dict):
      x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    else:
      x1, y1, x2, y2 = bbox
    
    box = np.array([x1, y1, x2, y2])

    if use_center_prompt:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        point_coords = np.array([[cx, cy]], dtype=np.float32)[None, ...]  
        point_labels = np.array([1], dtype=np.int32)[None, ...]         
    else:
        point_coords = None
        point_labels = None

    masks, scores, _ = self.predictor.predict(
      box=box[None, :], # (B, 4) => B: number of images, use None to make (4,) become (1,4)
      point_coords=point_coords, # (B, N, 2) => N: number of prompt point in each image
      point_labels=point_labels,
      multimask_output=multimask_output,
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx] # (H, W), contains only the values ​​0 or 1
    if self.enable_morph:
      mask = morphology_clean(mask)
    area = int(np.sum(mask))

    return {"mask": mask, "score": float(scores[best_idx]), "area": area}

  def segment_batch(
    self,
    image: Union[str, Path, np.ndarray, Image.Image],
    detections: List[Dict],
    pad_ratio=0.08,
  ):
    """Segment multiple YOLO detections efficiently"""
    if isinstance(image, (str, Path)):
      img_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
      img_array = np.array(image.convert("RGB"))
    else:
      img_array = image

    self.set_image(image=img_array)
    H, W = img_array.shape[:2]
    results = []

    for det in detections:
      try:
        x1, y1, x2, y2 = det["bbox"].values()
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, pad_ratio, (H, W))
        seg = self.segment_bbox((ex1, ey1, ex2, ey2))
        results.append({**det, **seg})
      except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        results.append({**det, "mask": None, "score": 0.0, "area": 0})
    
    return results
  

  @staticmethod
  def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5):
    if isinstance(image, Image.Image):
      image = np.array(image)
    colored = np.zeros_like(image)
    colored[mask] = color
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

  @staticmethod
  def mask_to_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(
      mask.astype(np.uint8), 
      cv2.RETR_EXTERNAL, 
      cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea) if contours else np.array([])
  

  @staticmethod
  def extract_masked_region(
    image: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (255, 255, 255)
  ) -> np.ndarray:
    if isinstance(image, Image.Image):
      image_array = np.array(image)
    else:
      image_array = image.copy()

    output = np.full_like(image_array, background_color)
    output[mask] = image_array[mask]

    return output
  

  @staticmethod
  def save_mask(
    self,
    mask: np.ndarray,
    output_path: Union[str, Path]
  ):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(output_path)



  

class GrabCutSegmenter:
  def __init__(self, cfg: Dict):
    self.cfg = cfg
    self.iterations = cfg.get('grabcut', {}).get('iterations', 5)
    logger.info("GrabCutSegmenter initialized (fallback)")

  def segment_bbox(
    self,
    image: Union[str, Path],
    bbox: Union[Dict, Tuple[float, float, float, float]]
  ) -> Dict:
    if isinstance(image, (str, Path)):
      img_array = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
      img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
      img_array = image

    if isinstance(bbox, dict):
      x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    else:
      x1, y1, x2, y2 = bbox
        
    rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    mask = np.zeros(img_array.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply grapcut
    cv2.grabCut(
      img_array, 
      mask,
      rect,
      bgd_model,
      fgd_model,
      cv2.GC_INIT_WITH_RECT
    )

    # Generate binary mask
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    area = np.sum(binary_mask)


    return {
      'mask': binary_mask.astype(bool),
      'score': 0.8, 
      'area': int(area),
      'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    }

