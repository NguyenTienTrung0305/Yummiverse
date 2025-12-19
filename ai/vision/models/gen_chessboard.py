import cv2
import numpy as np


width, height = 2480, 3508 
board = np.zeros((height, width), dtype=np.uint8) + 255 

square_size = 200 
rows = 6
cols = 9

# Vẽ các ô đen
start_x = (width - cols * square_size) // 2
start_y = (height - rows * square_size) // 2

for i in range(rows + 1): 
    for j in range(cols + 1):
        if (i + j) % 2 == 1:
            top_left = (start_x + j * square_size, start_y + i * square_size)
            bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
            cv2.rectangle(board, top_left, bottom_right, 0, -1)

cv2.imwrite("chessboard_print.png", board)
