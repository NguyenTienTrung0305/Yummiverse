import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VolumeEstimator:
  """
    Volume estimation:
      - Robust ArUco scale (median over multiple markers, outlier rejection)
      - Mask/contour area
      - Heuristic thickness by type/shape
      - Optional Monte Carlo for uncertainly (CI 95%) 
  """

  def __init__(self, cfg: Dict):
    self.cfg = cfg
    vol_cfg = cfg.get('volume_estimation', {})
    
    # ArUco
    self.aruco_dict_type = getattr(
      cv2.aruco,
      vol_cfg.get('aruco_dict', 'DICT_4X4_50')
    )
    self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
    self.aruco_params = cv2.aruco.DetectorParameters()
    
    self.reference_size = vol_cfg.get('reference_size_cm', 10.0)
    
    self.thickness_map = vol_cfg.get('thickness_cm', {})
    
    self.density_map = vol_cfg.get('density_g_per_cm', {})
    
    self.pixels_per_cm: Optional[float] = None
    logger.info("VolumeEstimator initialized")
    
    
  def detect_aruco_markers(
    self,
    image: Union[str, Path, np.ndarray, Image.Image]
  ):
    if isinstance(image, (str, Path)):
      img_array = cv2.imread(str(image))
    elif isinstance(image, Image.Image):
      img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
      img_array = image
      
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if corners is not None and len(corners) > 0:
      # Use first detected marker
      marker_corners = corners[0][0]
      
      # Calculate marker size in pixels
      # Average of all four sides
      side_lengths = []


