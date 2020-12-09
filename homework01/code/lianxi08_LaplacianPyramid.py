import cv2
import numpy as np
import math
#拉普拉斯金字塔
img = cv2.imread('picture.jpg')
lowest_res=32    # 最小分辨率
lowest_level = int(math.log(lowest_res, 2))
highest_res = img.shape[0] # 最大分辨率
highest_level = int(math.log(highest_res,2))

G_PYD = [] # 构建高斯金字塔
for level in range(highest_level,lowest_level-1,-1):
    if level==highest_level:
        G_PYD.append(img)
    else:
        img = cv2.pyrDown(img) # 下采样
        G_PYD.append(img)
G_PYD = list(reversed(G_PYD)) # 逆置
     # 构建拉普拉斯金字塔
L_PYD = []
L_PYD.append(G_PYD[0])
for idx in range(1,len(G_PYD)):
    # 下采样+上采样，减少信息量
    dim = cv2.pyrDown(G_PYD[idx])
    dim = cv2.pyrUp(dim)
        # 获取指定频带信息
    L_PYD.append(abs(G_PYD[idx]-dim))
img = cv2.imread('picture.jpg')
img = cv2.resize(img,(256,256))
for each in L_PYD:
    cv2.imshow(' ',each)
    cv2.waitKey(0)
