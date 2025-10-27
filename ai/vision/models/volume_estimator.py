from attr import dataclass
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Dataclass for calibration result
# ============================================================

@dataclass
class CalibrationResult:
    ppm: float                          # pixels per cm on metric plane
    H_img2metric: Optional[np.ndarray]  # 3x3 homography (image -> metric px)
    marker_id: Optional[int]
    marker_side_px: float
    side_px_std: float
    confidence: float                   # [0..1]
    corners_img: Optional[np.ndarray]   # (4,2) float32 corners (image px)

class VolumeEstimator:
  """
    Volume estimation:
      - Robust ArUco scale (median over multiple markers, outlier rejection)
      - Mask/contour area
      - Heuristic thickness by type/shape
      - Optional Monte Carlo for uncertainly (CI 95%)
      
    Estimate area_cm2 on a metric plane using:
      - If Homography available: transform contour points to metric pixels, area = contourArea / ppm^2
      - Else: fallback area_cm2 = area_pixels / ppm^2 (isotropic assumption)
    Volume = area_cm2 * thickness_cm
    Mass   = volume_cm3 * density_g_per_cm3 
  """

  def __init__(self, cfg: Dict):
    self.cfg = cfg
    vol_cfg = cfg.get('volume_estimation', {})
    
    self.thickness_map = vol_cfg.get('thickness_cm', {})
    self.density_map = vol_cfg.get('density_g_per_cm', {})
    
    self.enable_morph = vol_cfg.get('morphology', True)
    self.morph_kernel = vol_cfg.get('morph_kernel', 3)
    self.morph_iter = vol_cfg.get("morph_iter", 1)
    
    self.ppm_fallback = vol_cfg.get('ppm_fallback', 30.0)  # if no marker
    
  def estimate_from_mask(
    self,
    mask: np.ndarray,
    class_name: str = "default",
    thickness_cm: Optional[float] = None,
    density_g_per_cm3: Optional[float] = None,
    calib: Optional[CalibrationResult] = None,
  ):
    mask = _binary_mask(mask)
    cnt = _large_contour_area(mask)
    if cnt is None:
      return {
        "ok": False, 
        "reason": "empty_mask",
        "area_pixels": 0, 
        "area_cm2": 0.0,
        "volume_cm3": 0.0, 
        "mass_g": 0.0,
        "used_homography": bool(calib and calib.H_img2metric is not None),
        "calibration_confidence": float(calib.confidence if calib else 0.0),
        "uncertainty_method": "none"
      }


# ============================================================
# Helpers
# ============================================================
def _binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
  """
    Convert mask to uint8 {0,1} robustly:
    - bool -> {0,1}
    - uint8 0/1 -> keep
    - uint8 0/255 -> >0 => 1
    - float prob -> > thresh => 1
  """
  if mask.dtype == bool:
    return mask.astype(np.uint8)
  if mask.dtype == np.uint8:
    return (mask > 0).astype(np.uint8) if (mask > 1).any() else mask.copy()
  return (mask.astype(np.float32) > threshold).astype(np.uint8)

def _large_contour_area(mask: np.ndarray) -> float:
  cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not cnts:
    return None
  return max(cnts, key=cv2.contourArea)


# ============================================================
# OpenCV ArUco compatibility guards
# ============================================================
def _get_aruco_dict(dict_name: str):
  ad = getattr(cv2.aruco, dict_name, None)
  if ad is None:
    ad = cv2.aruco.DICT_4X4_50
  return cv2.aruco.getPredefinedDictionary(ad)


def _draw_marker_img(aruco_dict, marker_id: int, size_px: int) -> np.ndarray:
  if hasattr(cv2.aruco, 'generateImageMarker'):
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
  img = np.zeros((size_px, size_px), dtype=np.uint8)
  cv2.aruco.drawMarker(aruco_dict, marker_id, size_px, img, 1)
  return img

    

# ============================================================
# ArUco generator for printing
# ============================================================
def generate_aruco_marker(
    marker_id: int = 0, 
    size_cm: float = 10.0,
    dpi: int = 300,
    output_path: str | Path = "aruco_marker.png"
  ):
    """
    Args:
        size_cm: The size of marker when print on the paper (1 inch = 2.54 cm)
        dpi: Dots per inch for the generated image (resolution)
    """
    aruco_dict = _get_aruco_dict("DICT_4X4_50")
    size_px = int((size_cm / 2.54) * dpi)
    marker_img = _draw_marker_img(aruco_dict, marker_id, size_px)
    border = size_px // 10
    marker_with_border = cv2.copyMakeBorder(marker_img, border, border, border, border, cv2.BORDER_CONSTANT, 255)
    cv2.imwrite(str(output_path), marker_with_border)
    logger.info(f"Saved ArUco to {output_path} (print at {dpi} DPI for ~{size_cm} cm side)")
    
    
    