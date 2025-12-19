from math import pi
from attr import dataclass
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

@dataclass
class CameraIntrinsics:
    K: np.ndarray      # 3x3 camera intrinsic matrix
    dist: np.ndarray   # distortion coefficients (k,) usually (1,5) or (1,8)

class FoodVolumeEstimator:
    def __init__(
        self,
        cam: CameraIntrinsics,
        aruco_size_cm: float = 5.0,
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        aruco_dict = cv2.aruco.DICT_4X4_50,
        device = None,
        pixel_stride: int = 2, # cứ cách một pixel mới tính toán 3D một lần để tối ưu hóa tốc độ
        marker_sample_stride: int = 3
    ):
        self.cam = cam
        self.aruco_size_m = aruco_size_cm / 100.0
        self.pixel_stride = max(1, int(pixel_stride))
        self.marker_sample_stride = max(1, int(marker_sample_stride))
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        
        # Depth model (relative depth)
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model=depth_model,
            device=device
        )
        
        # densities (g/cm^3)
        self.food_densities = {
            "rice": 0.96,      
            "noodle": 0.80,
            "meat": 1.05,
            "fish": 1.05,
            "vegetable": 0.60,
            "soup": 1.00,
            "fruit": 0.70,
            "bread": 0.27,
            "egg": 1.03,
            "milk": 1.03,
            "default": 0.85
        }
        
    # -------------------------
    # Depth
    # -------------------------
    def estimate_depth_pred(self, bgr: np.ndarray):
        """
        Returns:
          depth_pred: (H,W) float32 (relative)
          depth_vis:  (H,W) float32 normalized for visualization only
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        out = self.depth_estimator(pil)
        
        # HuggingFace pipelines often return:
        #  - out["depth"]: PIL image
        #  - out["predicted_depth"]: torch tensor 
        pred = out.get("predicted_depth", None)
        if pred is not None:
            depth_pred = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            depth_pred = np.array(out["depth"]).astype(np.float32)
            
        # visualization only
        dmin, dmax = float(depth_pred.min()), float(depth_pred.max())
        depth_vis = (depth_pred - dmin) / (dmax - dmin + 1e-6)
        return depth_pred, depth_vis
    
    
    # -------------------------
    # ArUco + Plane
    # -------------------------
    def detect_aruco_pose_and_plane(self, bgr: np.ndarray):
        """
        Detects ArUco and estimates pose using camera intrinsics
        Returns:
          marker_corners (4,2), 
          marker_id (int), 
          rvec (3,), 
          tvec (3,)
          plane (n,d): plane equation in camera coords: n·X + d = 0, ||n||=1
          debug_image
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        debug = bgr.copy()
        if ids is None or len(ids) == 0:
            return None
        
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        
        # choose first marker
        marker_corners = corners[0][0].astype(np.float32)
        marker_id = int(ids[0][0])
        
        # pose estimation: OpenCV expects marker length in meters
        # Marker 3D points
        marker_points = np.array([
            [-self.aruco_size_m / 2,  self.aruco_size_m / 2, 0],
            [ self.aruco_size_m / 2,  self.aruco_size_m / 2, 0],
            [ self.aruco_size_m / 2, -self.aruco_size_m / 2, 0],
            [-self.aruco_size_m / 2, -self.aruco_size_m / 2, 0]
        ], dtype=np.float32)

        retval, rvec, tvec = cv2.solvePnP(
            marker_points,
            corners[0][0],
            self.cam.K,
            self.cam.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        # rvec cho biết trục xoay của marker trong hệ tọa độ camera
            # Hướng của rvec → trục xoay (vector)
            # Độ dài của rvec → góc xoay (rad)
        # Marker KHÔNG xoay quanh nhiều trục, mà luôn xoay quanh MỘT trục duy nhất (nhưng trục đó có thể nằm nghiêng giữa X–Y–Z)
        # rvec cho biết từ tu thế chuẩn, xoay theo trục nào, với góc bao nhiêu để đến được vị trí hiện tại của marker
        # Ví dụ mắt người là camera, 1 cái bảng là marker
            # + tư thế chuẩn là tư thế bảng đặt thẳng đứng vuông góc với mặt đất, mặt bảng hướng về phía người nhìn
            # + nếu bây giờ bảng nghiêng 45 độ về bên phải, thì rvec sẽ cho biết quay quanh trục nào, với góc bao nhiêu để từ tư thế chuẩn đến tư thế hiện tại
        rvec = rvec.reshape(3)
        
        # tvecs là tọa độ của tâm marker trong hệ tọa độ camera, đơn vị mét, trong hệ tọa độ của camera thì trục Oz hướng ra ngoài, Oy hướng lên trên, Ox hướng sang phải
        # ví dụ mình chụp góc 45-60 độ là góc hợp bởi Oz và mặt phẳng bàn mà marker nằm trên
        tvec = tvec.reshape(3)
        
        # draw axis for debug
        cv2.drawFrameAxes(debug, self.cam.K, self.cam.dist, rvec, tvec, self.aruco_size_m * 0.5)
        
        # Chuyển rvec sang ma trận xoay R (3x3)
        R, _ = cv2.Rodrigues(rvec)
        
        # Dựng vector pháp tuyến của mặt phẳng marker trong hệ tọa độ camera
            # => sử dụng vector pháp tuyến để dựng phương trính mặt phẳng và tính khoảng cách từ camera đến mặt phẳng
        # Trong hệ tọa độ marker, vector pháp tuyến là (0,0,1)
            # chuyển sang hệ tọa độ camera phải áp dụng phép xoay: n = R @ (0,0,1)
        n = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Tính vector pháp tuyến đơn vị
        # n = n / ||n||
        n = n / (np.linalg.norm(n) + 1e-9)
        
        # Viết pt mặt phẳng: n·X + d = 0
            # n: vector pháp tuyến đơn vị
            # X: điểm bất kỳ trên mặt phẳng
            # d: khoảng cách có dấu từ gốc camera đến mặt bàn
        d = -float(n.dot(tvec))
        
        return marker_corners, marker_id, rvec, tvec, (n.astype(np.float32), float(d)), debug
    
    
    # -------------------------
    # Geometry helpers
    # -------------------------
    def _ray_dirs_from_pixels(self, us: np.ndarray, vs: np.ndarray):
        """
           Tạo tia ray từ các tọa độ pixel trên ảnh
            + us và vs là hai mảng số (arrays) chứa tọa độ của các điểm trên ảnh
              - u (trong us) là tọa độ ngang
              - v (trong vs) là tọa độ dọc
            + giả sử ảnh 10x10
              - muốn lấy tọa độ ở cột 2 hàng 3, -> u=2, v=3
              
            Hướng làm:
            1. chuyển tọa độ pixel (u,v) sang hệ tọa độ chuẩn hóa (x,y)-hệ tọa độ của camera
            2. Tạo vector ray (x,y,1), nghĩa là:
                - Đi từ camera origin (0,0,0)
                - Theo hướng x (sang ngang)
                - Theo hướng y (lên xuống)
                - Theo hướng z (về phía trước)
        """
        fx = self.cam.K[0, 0]
        fy = self.cam.K[1, 1]
        cx = self.cam.K[0, 2]
        cy = self.cam.K[1, 2]
        x = (us - cx) / fx
        y = (vs - cy) / fy
        ones = np.ones_like(x, dtype=np.float32)
        rays = np.stack([x, y, ones], axis=-1).astype(np.float32)  # (...,3)
        return rays
        
    def _intersect_rays_with_plane(self, rays: np.ndarray, plane_n: np.ndarray, plane_d: float):
        """
        Tính ra mảng tọa độ các điểm giao giữa nhiều tia sáng từ camera và một mặt phẳng, tất cả trong hệ tọa độ camera
            rays: (...,3) directions from camera origin
            plane: n·X + d = 0
            t: hệ số kéo dài, cho biết đi bao xa dọc theo (rays) để chạm vào mặt phẳng
            => từ rays và plane, tìm điểm giao nhau giữa tia và mặt phẳng:
                + X = rays * t
                + Plane: n @ X + d = 0 = n @ (rays * t) + d = 0
                => t = -d / (n @ rays)
                 
        """
        denom = np.sum(rays * plane_n.reshape((1,1,3)) if rays.ndim==3 else rays * plane_n, axis=-1)
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom) 
        t = (-plane_d) / denom 
        X = rays * t[..., None]  # (...,3)
        return X.astype(np.float32)
    
    
    # -------------------------
    # Depth calibration: depth_pred -> Z_metric
    # -------------------------
    def calibrate_depth_to_metric_Z(
        self,
        depth_pred: np.ndarray,
        marker_corners: np.ndarray,
        plane_n: np.ndarray,
        plane_d: float,
    ):
        """
        Args:
            depth_pred (np.ndarray): HxW relative depth prediction
            marker_corners (np.ndarray): (4,2) corners of detected marker in image
            plane_n (np.ndarray): (3,) normal vector of the marker plane in camera coords
            plane_d (float): distance from camera origin to marker plane 

        Returns:
            _type_: _description_
        """
        H, W = depth_pred.shape[:2]
        
        # Build a mask for marker polygon
        marker_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(marker_mask, marker_corners.astype(np.int32), 255) # fill white inside marker
        
        # sample pixels inside marker
        ys, xs = np.where(marker_mask == 255)
        if len(xs) < 200:
            raise RuntimeError("Marker region too small for calibration. Ensure marker is visible and big enough")
        
        # stride sampling
        idx = np.arange(0, len(xs), self.marker_sample_stride)
        xs_s = xs[idx].astype(np.float32)
        ys_s = ys[idx].astype(np.float32)
        
        # rays
        rays = self._ray_dirs_from_pixels(xs_s, ys_s) # (N,3)
        
        # true intersection on plane
        pts = self._intersect_rays_with_plane(rays, plane_n, plane_d) # (N,3)
        Z_true = pts[:, 2].astype(np.float32)
        
        # predicted depth values
        d_pred = depth_pred[ys_s.astype(np.int32), xs_s.astype(np.int32)].astype(np.float32)
        
        # robust fit (remove outliers by MAD on residual after initial fit)
        # initial fit
        a0, b0 = np.polyfit(d_pred, Z_true, 1)
        Z_hat = a0 * d_pred + b0
        residuals = Z_true - Z_hat
        
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) + 1e-9
        keep = np.abs(residuals - med) < 3.5 * mad
        
        if np.sum(keep) < 50:
            a = float(np.median(Z_true) / (np.median(d_pred) + 1e-9))
            b = 0.0
            return a, b

        a, b = np.polyfit(d_pred[keep], Z_true[keep], 1)
        return float(a), float(b)
     
    # Tính toán Z_metric từ depth_pred đã hiệu chỉnh
    # Z_metric là bản đồ khoảng cách (depth map) từ camera đến tất cả các pixel trong ảnh   
    def depth_pred_to_Zmetric(self, depth_pred: np.ndarray, a: float, b: float):
        Z = a * depth_pred + b
        # clamp to positive reasonable values
        Z = np.clip(Z, 0.05, 10.0).astype(np.float32)  # 5cm to 10m
        return Z
    

    # -------------------------
    # Volume integration
    # -------------------------
    def get_density(self, class_name: str):
        s = (class_name or "").lower()
        for k, v in self.food_densities.items():
            if k != "default" and k in s:
                return v
        return self.food_densities["default"]
    
    def estimate_volume_mass_one(
        self,
        mask: np.ndarray,
        Z_metric: np.ndarray,
        plane_n: np.ndarray,
        plane_d: float,
        class_name: str = "default",
    ):
        """
        Compute volume (cm^3) and mass (g) for one binary mask.

        For each sampled pixel:
          - compute 3D surface point from Z_metric and intrinsics
          - height = signed distance to plane (clamped >=0)
          - footprint area on plane for that pixel via intersecting 4 corner rays with plane
          - accumulate volume += height * area

        Returns dict with area_cm2, height_cm_median, volume_cm3, mass_g
        """
        H, W = mask.shape[:2]
        assert Z_metric.shape[:2] == (H, W)

        stride = self.pixel_stride
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {
                "area_cm2": 0.0,
                "height_cm_median": 0.0,
                "volume_cm3": 0.0,
                "mass_g": 0.0
            }

        # downsample indices for speed
        sel = np.arange(0, len(xs), stride)
        xs = xs[sel].astype(np.float32)
        ys = ys[sel].astype(np.float32)

        # 1) food surface 3D points
        fx = self.cam.K[0, 0]
        fy = self.cam.K[1, 1]
        cx = self.cam.K[0, 2]
        cy = self.cam.K[1, 2]

        # deprojection: phục hồi tọa độ 3D từ (tọa độ pixel + depth)
        # tạo rays hoàn chỉnh, dùng Z_metric (khoảng cách thực tế) để kéo dài ray đến đúng vị trí vật thể
        Z = Z_metric[ys.astype(np.int32), xs.astype(np.int32)]
        X = (xs - cx) / fx * Z
        Y = (ys - cy) / fy * Z
        
        # P không phải rays – mà là mảng vector vị trí, cho biết cách di chuyển từ gốc (0,0,0) đến điểm pixel trong không gian (vị trí kết thúc của rays)
        P = np.stack([X, Y, Z], axis=-1).astype(np.float32)  # (N,3) 
        
        # 2) height = khoảng cách vuông góc từ điểm P đến mặt phẳng
        # plane: n·X + d = 0 => signed distance = P·n + d (since ||n||=1)
        # n @ P: hình chiếu của vector P gốc là gốc tọa độ (0,0,0) lên vector n (pháp tuyến mặt phẳng)
        # Khoảng các từ P đến mặt phẳng bằng khoảng cách tuyệt đối từ gốc tọa độ đến mặt phẳng trừ đi khoảng cách tuyệt đối từ gốc tọa độ đến P 
            # theo hướng pháp tuyến của mặt phẳng
        h_m = (P @ plane_n) + plane_d
        h_m = np.clip(h_m, 0.0, None).astype(np.float32)
        

        # 3) Tính diện tích thực tế của mỗi pixel trên mặt phẳng, mỗi ô vuông pixel được tạo thành từ 1 hình vuông có 4 góc
        # => Cần tính diện tích chiếu xuống mặt phẳng của hình vuông này
        # corners for each pixel: (u,v), (u+1,v), (u,v+1), (u+1,v+1)
        u0 = xs
        v0 = ys
        u1 = xs + 1.0
        v1 = ys + 1.0

        # rays for the 4 corners
        r00 = self._ray_dirs_from_pixels(u0, v0)  # (N,3)
        r10 = self._ray_dirs_from_pixels(u1, v0)
        r01 = self._ray_dirs_from_pixels(u0, v1)
        r11 = self._ray_dirs_from_pixels(u1, v1)

        # intersect each with plane to get quad points
        # (camera origin is 0, so X = t*ray)
        def intersect(r):
            denom = np.sum(r * plane_n, axis=-1)
            denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
            t = (-plane_d) / denom
            return r * t[:, None]

        p00 = intersect(r00)
        p10 = intersect(r10)
        p01 = intersect(r01)
        p11 = intersect(r11)

        # area per pixel on plane (m^2)
        # vectorized triangle areas
        # tri1: p00,p10,p11  tri2: p00,p11,p01
        a1 = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
        a2 = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
        area_m2 = (a1 + a2).astype(np.float32)

        # 4) volume (m^3), Volume = Σ (height × footprint_area)
        vol_m3 = np.sum(h_m * area_m2)

        # Also estimate area_cm2 on plane (mask footprint)
        area_plane_m2 = float(np.sum(area_m2))
        area_cm2 = area_plane_m2 * 1e4  # 1 m^2 = 1e4 cm^2

        # height stats
        height_cm_med = float(np.median(h_m) * 100.0)

        # convert to cm^3 (1 m^3 = 1e6 cm^3)
        volume_cm3 = float(vol_m3 * 1e6)

        density = self.get_density(class_name)  # g/cm^3
        mass_g = float(volume_cm3 * density)

        return {
            "area_cm2": area_cm2,
            "height_cm_median": height_cm_med,
            "volume_cm3": volume_cm3,
            "mass_g": mass_g,
            "density_g_cm3": density
        }
    
    
    def process(
        self,
        bgr: np.ndarray,
        detections,
        use_marker_plane: bool = True
    ):
        """
        detections: list of dict with at least:
          - "mask": binary numpy array HxW
          - "class_name": str

        Returns:
          updated_detections, debug_info
        """
        # 1) ArUco pose + plane
        pose = self.detect_aruco_pose_and_plane(bgr)
        if pose is None:
            raise RuntimeError("No ArUco marker detected. For best accuracy, marker is required")
        
        marker_corners, marker_id, rvec, tvec, (plane_n, plane_d), aruco_dbg = pose
        
        # 2) Depth prediction
        depth_pred, depth_vis = self.estimate_depth_pred(bgr)
        
        # Ensure same size
        H, W = bgr.shape[:2]
        if depth_pred.shape[:2] != (H, W):
            depth_pred = cv2.resize(depth_pred, (W, H), interpolation=cv2.INTER_LINEAR)
            depth_vis = cv2.resize(depth_vis, (W, H), interpolation=cv2.INTER_LINEAR)
            
        # 3) Calibrate depth_pred -> Z metric
        a, b = self.calibrate_depth_to_metric_Z(depth_pred, marker_corners, plane_n, plane_d)
        Z_metric = self.depth_pred_to_Zmetric(depth_pred, a, b)
        
        # 4) Per detection compute volume/mass
        updated = []
        for det in detections:
            mask = det.get("mask", None)
            if mask is None:
                det["error"] = "missing_mask"
                updated.append(det)
                continue

            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

            cls = det.get("class_name", "default")
            stats = self.estimate_volume_mass_one(mask, Z_metric, plane_n, plane_d, class_name=cls)

            det = dict(det)
            det.update(stats)
            det["aruco_id"] = marker_id
            det["depth_calib_a"] = a
            det["depth_calib_b"] = b
            det["error"] = None
            updated.append(det)

        debug = {
            "aruco_debug_bgr": aruco_dbg,
            "depth_vis_0_1": depth_vis,
            "Z_metric_m": Z_metric,
            "plane_n": plane_n,
            "plane_d": plane_d,
            "aruco_id": marker_id,
            "calib_a": a,
            "calib_b": b
        }
        return updated, debug
    
    
# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # !!! Replace these with  calibrated intrinsics !!!
    # MUST calibrate once and paste results here
    K = np.array([
        [1200.0,   0.0, 640.0],
        [  0.0, 1200.0, 360.0],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float32)
    dist = np.zeros((1, 5), dtype=np.float32)

    cam = CameraIntrinsics(K=K, dist=dist)

    estimator = FoodVolumeEstimator(
        cam=cam,
        aruco_size_cm=5.0,
        depth_model="depth-anything/Depth-Anything-V2-Small-hf",
        pixel_stride=2,          
        marker_sample_stride=3
    )

    # Load an image
    img = cv2.imread("food.jpg")  # BGR

    # Example detections: you need to provide masks from your segmentation model
    # mask should be HxW binary (0/255 or 0/1)
    detections = [
        {"class_name": "rice", "mask": cv2.imread("mask_rice.png", 0)},
        {"class_name": "meat", "mask": cv2.imread("mask_meat.png", 0)},
    ]

    results, dbg = estimator.process(img, detections)

    for r in results:
        print("\n===", r.get("class_name"))
        print("Area (cm^2):", r["area_cm2"])
        print("Height median (cm):", r["height_cm_median"])
        print("Volume (cm^3):", r["volume_cm3"])
        print("Mass (g):", r["mass_g"])
        print("Density (g/cm^3):", r["density_g_cm3"])