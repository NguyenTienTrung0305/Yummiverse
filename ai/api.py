from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import io

from vision.models.detection_models import FoodDetectionModel, load_config
from vision.models.freshness_classifier import FreshnessClassifier

app = FastAPI(title="Vision API for Food Detection")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"]
)


# load config
CONFIG_PATH = "ai/vision/config/config.yaml"
config = load_config(CONFIG_PATH)


# initialize models
detection_model = None
freshness_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# response model
class IngredientDetection(BaseModel):
  name_vi: str
  name_en: str
  confidence: float
  quantity: Optional[float] = None
  bbox: Dict[str, float]


class FreshnessAssessment(BaseModel):
  name_vi: str
  name_en: str
  freshness_level: str
  freshness_score: float
  is_usable: bool

class VisionAnalysisResponse(BaseModel):
  session_id: str
  timestamp: str
  detected_ingredients: List[IngredientDetection]
  freshness_assessments: List[FreshnessAssessment]
  image_url: Optional[str] = None

@app.op_event("startup")
async def load_models():
  global detection_model, freshness_model
  detection_model = FoodDetectionModel(config)
  
  # load best weights if available
  weights_path = Path("./models/weights/food_detection_best.pt")
  if weights_path.exists():
    detection_model.load_model(weights_path)
  else:
    detection_model.load_model(None)
  detection_model.load_class_names(config['paths']['detection_data'])


  freshness_model = FreshnessClassifier(
    backbone=config["freshness_model"]['name'],
    num_classes=3,
    pretrained=True,
  ).to(device)
  freshness_weights_path = Path("./models/weights/freshness_best.pth")
  if freshness_weights_path.exists():
    freshness_model.load_state_dict(torch.load(freshness_weights_path, map_location=device))
  freshness_model.eval()


# user upload image -> bytes -> read bytes
def image_to_pil(file_bytes: bytes) -> Image.Image:
  return Image.open(io.BytesIO(file_bytes)).convert("RGB")

