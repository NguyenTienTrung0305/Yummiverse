import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def load_config(path) -> Dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)

def parse_device(s: Optional[str]) -> Optional[str]:
  if not s or s.lower() == 'auto':
    return None
  return s

def expand_and_clamp_box(x1, y1, x2, y2, H, W, margin):
  w = x2 - x1
  h = y2 - y1
  dx = w * margin
  dy = h * margin
  nx1 = max(0, np.floor(x1-dx)/2)
  ny1 = max(0, np.floor(y1-dy)/2)
  nx2 = min(W, np.ceil(x1+dx)/2)
  ny2 = min(H, np.ceil(y2+dy)/2)
  return nx1, ny1, nx2, ny2

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
    self.detection_confi = inf_cfg.get('detection_confi', 0.4)
    self.detection_iou = inf_cfg.get('detection_iou', 0.5)
    self.crop_margin = inf_cfg.get('crop_margin', 0.1)

    self.datayaml_path = cfg.get('paths', {}).get('detection_data', 'data_detection.yaml')

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

    
    try:
      # get list class name from weight
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

  

  # prepare all training parameters (kwargs) for YOLOv8
  # this kw dict will be passed in when calling model.train(**kw)
  def _build_train_kwags(self):
    tdet = self.cfg.get("training", {}).get("detection", {})
    aug = self.cfg.get("augmentation", {}).get("detection", {})
    opt = self.cfg.get("optimizer", {}).get("detection", {})

    kw = {
      "epochs": tdet.get("epochs", 300),
      "batch": tdet.get("batch_size", 32),
      "imgsz": tdet.get("img_size", 640),
      "project": "runs/detection",
      "name": "food_detection",
      "patience": tdet.get("patience", 50),
      "device": self.device,

      # augmentation hypers
      "hsv_h": aug.get("hsv_h", 0.015),
      "hsv_s": aug.get("hsv_s", 0.7),
      "hsv_v": aug.get("hsv_v", 0.4),
      "degrees": aug.get("degrees", 15.0),
      "fliplr": aug.get("fliplr", 0.5),
      "mosaic": aug.get("mosaic", 1.0),

      # optional advanced
      "mixup": aug.get("mixup", 0.0),
      "copy_paste": aug.get("copy_paste", 0.0),

      # optimizer hypers 
      "lr0": opt.get("lr0", 0.01),
      "weight_decay": opt.get("weight_decay", 0.0005),
    }

    return kw
  

  def train_with_config(self):
    if self.model is None:
      self.load_model(None)
    self.load_class_names(data_yaml=self.datayaml_path)

    train_kwargs = self._build_train_kwags()

    results = self.model.train(
      data=str(self.datayaml_path),
      **train_kwargs,
    )

    # find best.pt file in save directory of results
    best_src = Path(results.save_dir) / "weights" / "best.pt"

    weight_dir = Path("weights")
    weight_dir.mkdir(parents=True, exist_ok=True)

    best_dst = weight_dir / "food_detection_best.pt"
    if best_src.exists():
      shutil.copy2(best_src, best_dst)
      return best_dst
    else:
      return None
    

  def predict(self, images, conf, iou, agnostic_nms=False,max_det=300, verbose=False,):
    """
      - Predict bbox/mask
      - Return dir already parsed 
      - Support batch
    """

    if self.model is None:
      raise RuntimeError("Model not loaded, call load_model() first")
    
    confi_if = (self.detection_confi if conf is None else conf)
    iou_if = (self.detection_iou if iou is None else iou)

    is_single = not isinstance(images, (list, tuple))
    img_list= [images] if is_single else list(images)

    # inference (predict of yolo)
    results = self.model(
      img_list,
      conf=confi_if,
      iou=iou_if,
      agnostic_nms=agnostic_nms,
      max_det=max_det,
      verbose=verbose
    )

    parsed_batch = []

    for r in results:
      dets = []
      boxes = getattr(r, "boxes", None)

      if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
          x1, y1, x2, y2 = xyxy[i].tolist()
          cid = clses[i]
          det = {
            "class_id": cid,
            "class_names": self.class_names.get(cid, f"class_{cid}"),
            "vietnamese_names": self.vietnamese_names.get(cid, ""),
            "confidence": confs[i],
            "bbox": {
              "x1": x1,
              "y1": y1,
              "x2": x2,
              "y2": y2,
            } 
          }

          dets.append(det)
      parsed_batch.append(dets)

    return parsed_batch[0] if is_single else parsed_batch 


  @staticmethod
  def _to_pil(image):
    if isinstance(image, Image.Image):
      return image
    if isinstance(image, (str, Path)):
      return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
      if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
      if image.shape[2] == 3:
        return Image.fromarray(image)
      if image.shape[2] == 4:
        return Image.fromarray(image[..., :3])
      
  def draw(self, image, detections, color=(0,255,0), width=2, font_path=None, font_size=14):
    pil = self._to_pil(image=image).copy()
    draw = ImageDraw.Draw(pil)

    try:
      font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
    except Exception:
      font = ImageFont.load_default()

    for d in detections:
      x1, y1 = d["bbox"]["x1"], d["bbox"]["y1"]
      x2, y2 = d["bbox"]["x2"], d["bbox"]["y2"]
      name = d.get("class_names", "obj")
      conf = d.get("confidence", 0.0)
      label = f"{name} {conf:.2f}"

      # draw border
      for w in range(width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=color)

      # draw white text and black bạckground
      tw, th = draw.textsize(label, font=font)
      draw.rectangle([x1, y1-th-2, x1+tw+4, y1], fill=(0,0,0))
      draw.text((x1+2,y1-th-2), label, fill=(255,255,255), font=font)

    return pil
  


  def crop_images(self, image, detections, out_dir, margin=None, min_size=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = self._to_pil(image=image)
    W, H = pil.size
    margin = self.crop_margin if margin is None else margin

    saved = []

    for i, d in enumerate(detections):
      x1, y1 = d["bbox"]["x1"], d["bbox"]["y1"]
      x2, y2 = d["bbox"]["x2"], d["bbox"]["y2"]
      ex1, ey1, ex2, ey2 = expand_and_clamp_box(x1, y1, x2, y2, W, H, margin=margin)

      if ex2 - ex1 < min_size or ey2 - ey1 < min_size:
        continue

      crop = pil.crop((ex1, ey1, ex2, ey2))

      cname = d.get("class_names", "obj")
      outp = out_dir / f"crop_{i:03d}_{cname}.jpg"
      crop.save(outp)
      saved.append(outp)
    return saved


if __name__ == "__main__":
  cfg = load_config("config.yaml")

  det = FoodDetectionModel(cfg=cfg)
  det.load_model(None)
  det.load_class_names(data_yaml=cfg["paths"]["detection_data"])

  best_path = det.train_with_config()

  test_img = ""
  dets = det.predict(test_img)

  drawn = det.draw(test_img, dets)
  Path("outputs").mkdir(exist_ok=True, parents=True)
  drawn.save("outputs/test_with_boxes.jpg", quality=95)

  crop_paths = det.crop_images(test_img, dets, out_dir="outputs/crops")
