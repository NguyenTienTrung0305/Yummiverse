import numpy as np
import cv2
import glob


images_path = 'D:/Code/Python/Yummiverse/ai/datasets/calibration_images/*.jpg' 


CHECKERBOARD = (9, 6) 


SQUARE_SIZE = 2.4 

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane.


objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE


images = glob.glob(images_path)

if len(images) < 10:
    print("Số lượng ảnh quá ít (<10), kết quả có thể không chính xác")

gray = None
for fname in images:
    img = cv2.imread(fname)
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)


    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        
    else:
        print(f"Không tìm thấy bàn cờ trong ảnh: {fname}")

cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n============== KẾT QUẢ  ==============")
    print("Độ lệch trung bình (Reprojection Error):", ret)
    print("(Nếu < 1.0 là Tốt, < 0.5 là Rất Tốt)\n")
    
    print("K (Intrinsic Matrix):")
    print("np.array([")
    print(f"    [{mtx[0,0]:.4f}, {mtx[0,1]:.4f}, {mtx[0,2]:.4f}],")
    print(f"    [{mtx[1,0]:.4f}, {mtx[1,1]:.4f}, {mtx[1,2]:.4f}],")
    print(f"    [{mtx[2,0]:.4f}, {mtx[2,1]:.4f}, {mtx[2,2]:.4f}]")
    print("], dtype=np.float32)")
    
    print("\ndist (Distortion Coefficients):")
    print("np.array([")
    print(f"    {dist[0].tolist()}")
    print("], dtype=np.float32)")
    print("==============================================================")
else:
    print("Không tìm thấy bàn cờ trong bất kỳ ảnh nào. Kiểm tra lại ánh sáng hoặc góc chụp")