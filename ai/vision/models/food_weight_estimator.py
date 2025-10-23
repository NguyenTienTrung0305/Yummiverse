from typing import Dict, Union, Any
from pathlib import Path
import yaml
import json
import math

class FoodWeightEstimator:
  """
    + JUST DEMO -> QUICK, BASIC
      - Want to be most accurate -> Fixed camera (distance 60cm)
                                 -> Ingredients must be placed horizontally or vertically, not diagonally.
    + IMPROVEMENT:
      - PCA rotation-invariant (GrabCut)
      - Segment Anything (SAM) or Grounding-DINO + SAM
      - Depth Map + Mask (DPT / MiDaS)
  """

  FOOD_PROPERTIES: Dict[str, Any] = {}
  _density_calibration: Dict[str, float] = {}


  PROPERTIES_FILE = Path("../food_properties.yaml")
  CALIBRATION_FILE = Path("../calibration.json")

  @classmethod
  def load_properties(cls, path: str | Path = None):
    """ Call load_properties() before estimate_weight()"""
    file_path = Path(path or cls.PROPERTIES_FILE)
    if not file_path.exists():
      raise FileNotFoundError(f"Food properties file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
      cls.FOOD_PROPERTIES = yaml.safe_load(f)

  @classmethod
  def load_calibration(cls, path: str | Path = None):
    """Load previously learned density calibrations (JSON)."""
    file_path = Path(path or cls.CALIBRATION_FILE)
    if not file_path.exists():
      print("[INFO] No calibration file found. Starting fresh.")
      return
    with open(file_path, "r", encoding="utf-8") as f:
      cls._density_calibration = json.load(f)

  @classmethod
  def save_calibration(cls, path: str | Path = None):
    """Persist calibration to JSON file."""
    file_path = Path(path or cls.CALIBRATION_FILE)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(cls._density_calibration, f, indent=2, ensure_ascii=False)



  @classmethod
  def estimate_weight(
    cls,
    bbox: Dict[str, float],
    img_width,
    img_height,
    class_name,
    confidence,
    distance_factor: float = 1.0
  ) -> Dict[str, Union[float, str]]:
    box_width = bbox["x2"] - bbox["x1"]
    box_height = bbox["y2"] - bbox["y1"]

    area_ratio = (box_width * box_height) / (img_width * img_height)
    aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height),1)

    if class_name not in cls.FOOD_PROPERTIES:
      return cls._estimate_generic(area_ratio)

    props = cls.FOOD_PROPERTIES[class_name]

    density = cls._density_calibration.get(class_name, props['density_factor'])

    # 1. Physical size factor (scale by volume)
    size_factor = (max(area_ratio, 0.01) ** 1.5) * 100

    # 2. Thickness / Shape
    depth_factor = cls._calculate_depth_factor(aspect_ratio, props['shape_factor'])

    # 3. Adjust according to camera distance
    distance_adjustment = 1.0 / (distance_factor ** 2)

    # 4. Estimate weight
    estimated_weight = (
      props['base_weight'] *
      density *  
      size_factor * 
      depth_factor * 
      confidence * 
      distance_adjustment
    ) / 100.0

    # 5. Confine
    estimated_weight = max(
      props['min_weight'],
      min(estimated_weight, props['max_weight'])
    )

    # 6. Uncertainty & Confidence Level
    uncertainty = cls._calculate_uncertainty(area_ratio, confidence)
    min_weight = estimated_weight * (1 - uncertainty)
    max_weight = estimated_weight * (1 + uncertainty)


    return {
      'weight': round(estimated_weight, 1),
      'min_weight': round(min_weight, 1),
      'max_weight': round(max_weight, 1),
      'unit': 'gram'
    }
  
  
  @staticmethod
  def _calculate_depth_factor(aspect_ratio, shape_factor) -> float:
    aspect_ratio = min(aspect_ratio, 5.0)
    depth_multiplier = 1.2 - 0.25 * math.log1p(aspect_ratio - 1)
    depth_multiplier = max(0.6, min(depth_multiplier, 1.2))
    return depth_multiplier * shape_factor
  
  @staticmethod
  def _calculate_uncertainty(area_ratio: float, confidence: float) -> float:
    base = 0.25 if area_ratio < 0.03 or area_ratio > 0.4 else 0.15
    conf_penalty = (1 - confidence) * 0.25
    return min(0.4, base + conf_penalty)
  
  @staticmethod
  def _estimate_generic(area_ratio):
    base = 100 + 1500 * min(area_ratio, 1.0)
    return {
      "weight": round(base, 1),
      "min_weight": round(base * 0.5, 1),
      "max_weight": round(base * 1.5, 1),
      "unit": "gram",
    }
  

  @classmethod
  def update_density_from_sample(cls, class_name: str, predicted_weight: float, actual_weight: float):
    """ 
      Update density_factor based on real measurement (EMA smoothing) 
    """
    if class_name not in cls.FOOD_PROPERTIES:
      return

    # Giá trị hiện tại
    old = cls._density_calibration.get(
      class_name,
      cls.FOOD_PROPERTIES[class_name]["density_factor"]
    )
    ratio = actual_weight / max(predicted_weight, 1e-3)
    new = 0.9 * old + 0.1 * (old * ratio)  # EMA smoothing

    cls._density_calibration[class_name] = round(new, 4)
    print(f"[LEARN] Updated {class_name}: {old:.3f} → {new:.3f}")

  @classmethod
  def learn_and_save(cls, class_name: str, predicted_weight: float, actual_weight: float):
    cls.update_density_from_sample(class_name, predicted_weight, actual_weight)
    cls.save_calibration()

if __name__ == "__main__":
  FoodWeightEstimator.load_properties()
  FoodWeightEstimator.load_calibration()

  bbox = {"x1": 120, "y1": 100, "x2": 450, "y2": 380}
  result = FoodWeightEstimator.estimate_weight(
      bbox=bbox,
      img_width=1920,
      img_height=1080,
      class_name="apple",
      confidence=0.93
  )

  predicted_weight = result["weight"]
  print(f"[PREDICT] apple: {predicted_weight} g (range {result['min_weight']}-{result['max_weight']})")


  actual_weight = 230.0 

  FoodWeightEstimator.learn_and_save(
      class_name="apple",
      predicted_weight=predicted_weight,
      actual_weight=actual_weight
  )
