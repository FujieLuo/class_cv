import cv2
import numpy as np

image_path = 'picture.jpg'  # 读取待变换图像
img = cv2.imread(image_path)
height, width, channel = img.shape
src_point = np.array([[0, 0, height - 1], [0, width - 1, 0], [1, 1, 1]])  # 原始图像选三个点
dst_point = np.array([[100, 50, 800], [50, 800, 300], [1, 1, 1]])  # 变换后图像上对应的点
T = np.dot(dst_point, np.linalg.inv(src_point))  # 求取仿射变换矩阵

affine_img = np.zeros((height, width, 3), np.uint8)  # 创建变换后图像

for x in range(height):
    for y in range(width):
        src_image_point = np.array([[x, y, 1]])
        src_image_point_transpose = np.transpose(src_image_point)
        # print( src_image_point_transpose )
        a = np.dot(T, src_image_point_transpose)
        new_x, new_y = a[:2]
        affine_img[int(new_x)][int(new_y)][:] = img[x][y]

cv2.imshow('orign_image', img)
cv2.imshow('AffineImage', affine_img)
cv2.waitKey(0)
