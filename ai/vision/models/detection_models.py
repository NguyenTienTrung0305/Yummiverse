import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ==========================
# Utils
# ==========================
def load_config(path: str | Path) -> Dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def parse_device(s: Optional[str]) -> Optional[str]:
  if not s or str(s).lower() == "auto":
    return None
  return str(s)


def expand_and_clamp_box(x1, y1, x2, y2, W, H, margin=0.1):
  w, h = x2 - x1, y2 - y1
  dx, dy = w * margin, h * margin
  nx1 = max(0, int(np.floor(x1 - dx)))
  ny1 = max(0, int(np.floor(y1 - dy)))
  nx2 = min(W, int(np.ceil(x2 + dx)))
  ny2 = min(H, int(np.ceil(y2 + dy)))
  return nx1, ny1, nx2, ny2


# ==========================
# Main model class (SEGMENTATION)
# ==========================
class FoodSegmentationModel:
  """
  Model segmentation cho nguyên liệu (YOLOv8-seg)
  Hỗ trợ:
    - Train với config
    - Predict bbox + mask
    - Vẽ bbox/mask
    - Crop ảnh theo bbox
    - Tính area từ mask (phục vụ ước lượng khối lượng)
  """

  def __init__(self, cfg: Dict[str, Any]):
    self.cfg = cfg
    self.model: Optional[YOLO] = None

    self.model_name = (
      cfg.get("detection_model", {}).get("name", "yolov8s-seg")
    )
    self.device: Optional[str] = parse_device(
      cfg.get("training", {}).get("device", "auto")
    )

    self.class_names: Dict[int, str] = {}
    self.vietnamese_names: Dict[int, str] = {}

    # inference config
    inf_cfg = cfg.get("inference", {}) or {}
    self.detection_confi = inf_cfg.get("detection_confi", 0.4)
    self.detection_iou = inf_cfg.get("detection_iou", 0.5)
    self.crop_margin = inf_cfg.get("crop_margin", 0.1)

    self.datayaml_path = cfg.get("paths", {}).get(
      "detection_data", "data_detection.yaml"
    )


  # --------------------------
  # Load model & class names
  # --------------------------
  def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
    """
    Load YOLOv8-seg model
    - Nếu truyền model_path: dùng weight đó
    - Nếu không: dùng <model_name>.pt (vd: yolov8s-seg.pt)
    """
    if model_path:
      model_path = Path(model_path)
      if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
      self.model = YOLO(str(model_path))
    else:
      self.model = YOLO(f"{self.model_name}.pt")

    if self.model is not None and self.device is not None:
      try:
        self.model.to(self.device)
      except Exception as e:
        print(f"[WARN] Could not set device {self.device}: {e}")

    try:
      if self.model is not None:
        names = self.model.names
        if isinstance(names, dict) and names:
          self.class_names = {int(k): v for k, v in names.items()}
    except Exception:
      pass


  def load_class_names(self, data_yaml: str | Path | None = None) -> None:
    """
    Load class_names & vietnamese_names từ file data_detection.yaml
    """
    path = Path(data_yaml or self.datayaml_path)
    if not path.exists():
      print(f"[WARN] data_detection.yaml not found at {path}")
      return

    with open(path, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f) or {}

    names = data.get("names", {})
    if isinstance(names, list):
      self.class_names = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
      self.class_names = {int(k): v for k, v in names.items()}

    vn_names = data.get("vietnamese_names", {})
    if isinstance(vn_names, list):
      self.vietnamese_names = {i: n for i, n in enumerate(vn_names)}
    elif isinstance(vn_names, dict):
      self.vietnamese_names = {int(k): v for k, v in vn_names.items()}
      

  # --------------------------
  # Build train kwargs
  # --------------------------
  def _build_train_kwargs(self) -> Dict[str, Any]:
    tdet = self.cfg.get("training", {}).get("detection", {})
    aug = self.cfg.get("augmentation", {}).get("detection", {})
    opt = self.cfg.get("optimizer", {}).get("detection", {})

    kw = {
      "epochs": tdet.get("epochs", 300),
      "batch": tdet.get("batch_size", 16),
      "imgsz": tdet.get("img_size", 640),
      "project": "runs/segmentation",
      "name": "food_segmentation",
      "patience": tdet.get("patience", 50),
      "device": self.device,

      # augmentation
      "hsv_h": aug.get("hsv_h", 0.007),
      "hsv_s": aug.get("hsv_s", 0.5),
      "hsv_v": aug.get("hsv_v", 0.3),
      "degrees": aug.get("degrees", 15.0),
      "fliplr": aug.get("fliplr", 0.5),
      "mosaic": aug.get("mosaic", 0.9),
      "mixup": aug.get("mixup", 0.0),
      "copy_paste": aug.get("copy_paste", 0.3),

      # optimizer
      "lr0": opt.get("lr0", 0.009),
      "weight_decay": opt.get("weight_decay", 0.0005),
    }

    return kw


  # --------------------------
  # Train
  # --------------------------
  def train_with_config(self) -> Optional[Path]:
    """
    Train YOLOv8-seg với config + data_detection.yaml
    Trả về path tới best.pt nếu thành công
    """
    if self.model is None:
      self.load_model(None)
    self.load_class_names(data_yaml=self.datayaml_path)

    train_kwargs = self._build_train_kwargs()

    assert self.model is not None
    results = self.model.train(
        data=str(self.datayaml_path),
        **train_kwargs,
    )
    best_src = Path(results.save_dir) / "weights" / "best.pt"

    weight_dir = Path("weights")
    weight_dir.mkdir(parents=True, exist_ok=True)
    best_dst = weight_dir / "food_segmentation_best.pt"
    if best_src.exists():
      shutil.copy2(best_src, best_dst)
      print(f"[INFO] Best model saved to: {best_dst}")
      return best_dst
    else:
      print("[WARN] best.pt not found in results directory")
      return None


  # --------------------------
  # Predict (bbox + mask)
  # --------------------------
  def predict(
    self,
    images: Union[str, Path, Image.Image, np.ndarray, Sequence],
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    agnostic_nms: bool = False,
    max_det: int = 300,
    verbose: bool = False,
  ):
    """
    - Predict bbox + mask
    - Hỗ trợ batch
    - Trả về list detections đã parse:
      {
        "class_id": int,
        "class_name": str,
        "vietnamese_name": str,
        "confidence": float,
        "bbox": {x1,y1,x2,y2},
        "mask": np.ndarray | None   # (H, W) với 0/1
      }
    """
    if self.model is None:
      raise RuntimeError("Model not loaded, call load_model() first")

    confi_if = self.detection_confi if conf is None else conf
    iou_if = self.detection_iou if iou is None else iou

    is_single = not isinstance(images, (list, tuple))
    img_list = [images] if is_single else list(images)

    results = self.model(
      img_list,
      conf=confi_if,
      iou=iou_if,
      agnostic_nms=agnostic_nms,
      max_det=max_det,
      verbose=verbose,
    )

    parsed_batch: List[List[Dict[str, Any]]] = []

    for r in results:
      dets: List[Dict[str, Any]] = []
      boxes = getattr(r, "boxes", None)
      masks = getattr(r, "masks", None)

      if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        if masks is not None:
          mask_arr = masks.data.cpu().numpy()  # (N, H, W)
        else:
          mask_arr = []

        for i in range(len(xyxy)):
          x1, y1, x2, y2 = xyxy[i].tolist()
          cid = clses[i]
          det = {
            "class_id": cid,
            "class_name": self.class_names.get(cid, f"class_{cid}"),
            "vietnamese_name": self.vietnamese_names.get(cid, ""),
            "confidence": float(confs[i]),
            "bbox": {
              "x1": float(x1),
              "y1": float(y1),
              "x2": float(x2),
              "y2": float(y2),
            },
            "mask": mask_arr[i] if len(mask_arr) > 0 else None,
          }
          dets.append(det)

        parsed_batch.append(dets)

    return parsed_batch[0] if is_single else parsed_batch


  # --------------------------
  # PIL helper
  # --------------------------
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
    raise TypeError("Unsupported image type for _to_pil")

  # --------------------------
    # Draw bbox + (optional) mask
    # --------------------------

  def draw(
    self,
    image,
    detections: List[Dict[str, Any]],
    color=(0, 255, 0),
    width: int = 2,
    font_path: Optional[Union[str, Path]] = None,
    font_size: int = 14,
    draw_masks: bool = False,
    mask_alpha: float = 0.35,
  ):
    """
    Vẽ bbox + (optional) mask lên ảnh
    """
    pil = self._to_pil(image=image).copy()
    draw = ImageDraw.Draw(pil)

    try:
      font = (
        ImageFont.truetype(str(font_path), font_size)
        if font_path
        else ImageFont.load_default()
      )
    except Exception:
      font = ImageFont.load_default()

    W, H = pil.size

    if draw_masks:
      overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
      overlay_draw = ImageDraw.Draw(overlay)

    for d in detections:
      x1, y1 = d["bbox"]["x1"], d["bbox"]["y1"]
      x2, y2 = d["bbox"]["x2"], d["bbox"]["y2"]
      name = d.get("class_name", "obj")
      conf = d.get("confidence", 0.0)
      label = f"{name} {conf:.2f}"

      # vẽ mask nếu có
      if draw_masks and d.get("mask") is not None:
        mask = d["mask"].astype(bool)
      
        # convert mask thành polygon bằng cách lấy contour đơn giản
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
          # bounding rect của mask
          min_x, max_x = xs.min(), xs.max()
          min_y, max_y = ys.min(), ys.max()
          overlay_draw.rectangle(
            [min_x, min_y, max_x, max_y],
            fill=(255, 0, 0, int(255 * mask_alpha)),
          )

        # vẽ bbox
        for w in range(width):
          draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)

          # label
          left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
          tw, th = right - left, bottom - top
          draw.rectangle(
            [x1, y1 - th - 2, x1 + tw + 4, y1],
            fill=(0, 0, 0),
          )
          draw.text(
            (x1 + 2, y1 - th - 2),
            label,
            fill=(255, 255, 255),
            font=font,
          )

    if draw_masks:
      pil = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")

    return pil

  # --------------------------
  # Crop theo bbox
  # --------------------------
  def crop_images(
    self,
    image,
    detections: List[Dict[str, Any]],
    out_dir: Union[str, Path],
    margin: Optional[float] = None,
    min_size: int = 8,
  ) -> List[Path]:
    """
    Crop từng object theo bbox, trả về list path đã lưu
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = self._to_pil(image=image)
    W, H = pil.size
    margin = self.crop_margin if margin is None else margin
    saved: List[Path] = []

    for i, d in enumerate(detections):
      x1, y1 = d["bbox"]["x1"], d["bbox"]["y1"]
      x2, y2 = d["bbox"]["x2"], d["bbox"]["y2"]
      ex1, ey1, ex2, ey2 = expand_and_clamp_box(
        x1, y1, x2, y2, W, H, margin=margin
      )
      if ex2 - ex1 < min_size or ey2 - ey1 < min_size:
        continue

      crop = pil.crop((ex1, ey1, ex2, ey2))

      cname = d.get("class_name", "obj")
      outp = out_dir / f"crop_{i:03d}_{cname}.jpg"
      crop.save(outp, quality=95)
      saved.append(outp)

    return saved


  # --------------------------
  # Mask area
  # --------------------------
  @staticmethod
  def mask_area(mask: Optional[np.ndarray]) -> int:
    """
    Diện tích (số pixel) của mask
    """
    if mask is None:
      return 0
    # mask dạng 0/1 (hoặc 0.0/1.0)
    return int(mask.astype(bool).sum())


if __name__ == "__main__":
    cfg = load_config("../config/config.yaml")

    seg = FoodSegmentationModel(cfg=cfg)
    seg.load_model(None)  
    seg.load_class_names(data_yaml=cfg["paths"]["detection_data"])

  
    best_path = seg.train_with_config()

    test_img = "path/to/your/test_image.jpg"  

    dets = seg.predict(test_img, conf=None, iou=None)
    print("Detections:", dets)

    drawn = seg.draw(test_img, dets, draw_masks=True)
    Path("outputs").mkdir(exist_ok=True, parents=True)
    drawn.save("outputs/test_with_boxes_masks.jpg", quality=95)

    crop_paths = seg.crop_images(test_img, dets, out_dir="outputs/crops")
    print("Crops saved:", crop_paths)
