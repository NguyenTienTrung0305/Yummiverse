import requests
import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import cv2


class NutritionAPIClient:
  """Client để gửi kết quả volume estimation sang Nutrition API"""
  
  def __init__(self, base_url: str = "http://localhost:3000"):
    self.base_url = base_url
    self.api_url = f"{self.base_url}/api/nutrition"
    
  def send_volume_nutrition_data(
    self,
    session_id: str,
    detections:  List[Dict[str, Any]]
  ):
    serializable_detections = []
    for det in detections:
      det_copy = det.copy()
      
      if "mask" in det_copy:
        del det_copy["mask"]
      
      for key, value in det_copy.items():
        if isinstance(value, np.ndarray):
          det_copy[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
          det_copy[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
          det_copy[key] = int(value)
        elif isinstance(value, dict):
          # Handle nested dicts (like bbox)
          det_copy[key] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in value.items()
          }
            
      serializable_detections.append(det_copy)
    
    payload = {
      "session_id": session_id,
      "detections": serializable_detections
    }
    
    
    response = requests.post(
      f"{self.api_url}/ingredients/volume-nutrition",
      json=payload,
      headers={"Content-Type": "application/json"}
    )
    
    response.raise_for_status()
    return response.json()
  
  
  def get_llm_input(self, session_id: str) -> Dict[str, Any]:
    """Lấy formatted LLM input cho session đã có sẵn"""
    response = requests.get(
      f"{self.api_url}/ingredients/{session_id}/llm-input"
    )
    response.raise_for_status()
    return response.json()
  

  def save_llm_input_to_file(
    self, 
    llm_input: Dict[str, Any], 
    output_path: str
  ) -> None:
    """Lưu LLM input ra file để dễ sử dụng"""
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(llm_input, f, ensure_ascii=False, indent=2)
    print(f"LLM input saved to: {output_path}")
    
  


def integrate_volume_estimation_with_nutrition(
  bgr_image: np.ndarray,
  segmentation_model,
  volume_estimator,
  nutrition_client: NutritionAPIClient,
  session_id: str,
  save_debug: bool = True,
  output_dir: str = "outputs"
):
  """
  Pipeline hoàn chỉnh: 
  1. Detect & segment ingredients
  2. Estimate volume & mass
  3. Enrich with nutrition data
  4. Format for LLM
  """
  # 1. Chuyển thành Path object
  out_path = Path(output_dir)
  
  if save_debug:
      out_path.mkdir(parents=True, exist_ok=True)

  detections = segmentation_model.predict(bgr_image, verbose=False)
  
  if save_debug:
    vis = segmentation_model.draw(
      bgr_image, 
      detections, 
      draw_masks=True,
      mask_alpha=0.4
    )

    vis.save(str(out_path / "01_segmentation.jpg"), quality=95)
    
  volume_results, debug_info = volume_estimator.process(
    bgr_image, 
    detections,
    use_marker_plane=True
  )

  if save_debug:
    aruco_dbg = debug_info["aruco_debug_bgr"]

    cv2.imwrite(
      str(out_path / "02_aruco_detection.jpg"),
      aruco_dbg
    )
        
    depth_vis = (debug_info["depth_vis_0_1"] * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    cv2.imwrite(
      str(out_path / "03_depth_estimation.jpg"),
      depth_vis
    )
    
  
  try:
    api_response = nutrition_client.send_volume_nutrition_data(
      session_id=session_id,
      detections=volume_results
    )
    
    if api_response.get("success"):
      print("Nutrition data successfully integrated")
    else:
      print("Failed to integrate nutrition data:", api_response)
  except Exception as e:
    print(f"Failed to connect to API: {e}")
    return {
      "success": False,
      "error": str(e),
      "detections": volume_results,
      "debug_info": debug_info
    }
    
  llm_input = api_response.get("llm_input")
    

  if save_debug and llm_input:
    nutrition_client.save_llm_input_to_file(
      llm_input,
      str(out_path / "llm_input.json") 
    )
    
      
  return {
    "success": True,
    "session_id": session_id,
    "api_response": api_response,
    "llm_input": llm_input,
    "debug_info": debug_info,
    "output_dir": str(out_path)
  }
    


if __name__ == "__main__":
  from vision.models.detection_models import FoodSegmentationModel, load_config
  from vision.models.food_volume_estimator import FoodVolumeEstimator, CameraIntrinsics
  import uuid
    
  # Load models
  cfg = load_config("./vision/config/config.yaml")
    
  seg_model = FoodSegmentationModel(cfg=cfg)
  seg_model.load_model("D:/Code/Python/Yummiverse/ai/vision/runs/best.pt")
  seg_model.load_class_names()
    
  # Camera intrinsics (MUST calibrate your camera!)
  K = np.array([
      [1075.9064, 0.0000, 590.9764],
      [0.0000, 1081.0149, 592.7255],
      [0.0000, 0.0000, 1.0000]
  ], dtype=np.float32)
  dist = np.array([
    [-0.40075333, 2.96821205, -0.00185262, -0.00319792, -8.76258945]
  ], dtype=np.float32) 
    
  cam = CameraIntrinsics(K=K, dist=dist)
  volume_est = FoodVolumeEstimator(
    cam=cam,
    aruco_size_cm=5.0,
    depth_model="depth-anything/Depth-Anything-V2-Small-hf",
    pixel_stride=2,
    marker_sample_stride=3
  )
    
  # API client
  api_client = NutritionAPIClient(base_url="http://localhost:3000")
    
  # Process image
  img = cv2.imread("C:/Users/ADMIN/Downloads/test_img3.jpg")
  session_id = str(uuid.uuid4())
    
  results = integrate_volume_estimation_with_nutrition(
    bgr_image=img,
    segmentation_model=seg_model,
    volume_estimator=volume_est,
    nutrition_client=api_client,
    session_id=session_id,
    save_debug=True,
    output_dir=f"outputs/{session_id}"
  )
    
  if results["success"]:
    print("\n Analysis complete!")
    print("\nLLM Input Preview:")
    print(json.dumps(
      results["llm_input"]["summary"], 
      indent=2, 
      ensure_ascii=False
    ))
  else:
    print(f"\n Error: {results.get('error')}")