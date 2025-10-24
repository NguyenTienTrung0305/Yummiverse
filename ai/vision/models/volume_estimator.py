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
    dict_name = vol_cfg.get("aruco_dict", 'DICT_4X4_50')
    self.reference_size_cm = vol_cfg.get('reference_size_cm', 10.0)

    self._aruco = cv2.aruco
    self._aruco_dict = self._aruco.getPredefinedDictionary(getattr(self._aruco, dict_name))




