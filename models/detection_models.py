import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def parse_device(s: Optional[str]) -> Optional[str]:
  if not s or s.lower() == 'auto':
    return None
  return s

class FoodDetectionModel:
  def __init__(self, cfg: Dict[str, Any]):
    self.cfg = cfg
    self.model: Optional[YOLO] = None

    self.model_name = cfg.get('detection_model', {}).get('name', 'yolov8s')
    self.device: Optional[str] = parse_device(cfg.get("training", {}).get("device", "auto"))

    self.class_names: Dict[int, str] = {}
    self.vietnamese_names: Dict[int, str] = {}

    # inference config 
    inf_cfg = cfg.get('inference', {}) or {}
    self.detection_conf = inf_cfg.get('detection_conf', 0.4)
    self.detection_iou = inf_cfg.get('detection_iou', 0.5)
    self.crop_margin = inf_cfg.get('crop_margin', 0.1)

    self.datayaml_path = inf_cfg.get('data', {}).get('detection_data', 'data_detection.yaml')

  def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
    if model_path:
      model_path = Path(model_path)
      if not model_path.exists():
        raise FileNotFoundError(f"Model file not found")
      
      self.model = YOLO(str(model_path))
    
    else:
      self.model = YOLO(f"{self.model_name}.pt")

    if self.device is not None:
      try:
        self.model.to(self.device)
      except Exception as e:
        print(f"Could not set device")

    
    # get list class name from weight
    try:
      names = self.model.names
      if isinstance(names, dict) and names:
        self.class_names = {int(k): v for k,v in names.items()}
    except Exception:
      pass

  
  def load_class_names(self, data_yaml: str | Path | None) -> None:
    path = Path(data_yaml or self.datayaml_path)
    if not path.exists():
      return
    
    with open(path, 'r', encoding='utf8') as f:
      data = yaml.safe_load(f) or {}
    
    names = data.get('names', {})
    if isinstance(names, list):
      self.class_names = {i:n for i,n in enumerate(names)}
    elif isinstance(names, dict):
      self.class_names = {int(k): v for k,v in names.items()}

    vn_names = data.get('vietnamese_names', {})
    if isinstance(vn_names, list):
      self.vietnamese_names = {i:n for i,n in enumerate(vn_names)}
    elif isinstance(names, dict):
      self.vietnamese_names = {int(k): v for k,v in vn_names.items()}
