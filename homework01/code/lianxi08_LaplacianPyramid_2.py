import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取原始图像
img = cv2.imread('picture.jpg')
img = cv2.resize(img,(512,256))
r1 = cv2.pyrDown(img) # 图像向下取样
r2 = cv2.pyrUp(r1) # 图像向上取样
LapPyr0 = img - r2  # 拉普拉斯第0层

r3 = cv2.pyrDown(r1) # 图像向下取样
r4 = cv2.pyrUp(r3) # 图像向上取样
LapPyr1 = r1 - r4  # 拉普拉斯第1层

r5 = cv2.pyrDown(r3) # 图像向下取样
r6 = cv2.pyrUp(r5) # 图像向上取样
LapPyr2 = r3 - r6  # 拉普拉斯第2层

r7 = cv2.pyrDown(r5) # 图像向下取样
r8 = cv2.pyrUp(r7) # 图像向上取样
LapPyr3 = r5 - r8  # 拉普拉斯第3层

cv2.imshow('original', img)
cv2.imshow('LapPyr0', LapPyr0)
cv2.imshow('LapPyr1', LapPyr1)
cv2.imshow('LapPyr2', LapPyr2)
cv2.imshow('LapPyr3', LapPyr3)
cv2.waitKey()
cv2.destroyAllWindows()