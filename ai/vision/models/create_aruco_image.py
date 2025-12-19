import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


img = cv2.aruco.generateImageMarker(aruco_dict, id=0, sidePixels=1000, borderBits=1)

cv2.imwrite("aruco_id0.png", img)