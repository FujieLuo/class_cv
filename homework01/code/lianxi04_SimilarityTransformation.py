import cv2
import numpy as np
import math
img = cv2.imread('picture.jpg')#读取图片成张量，左上为原点
rows = img.shape[0] #取图片的行数，即高度
cols = img.shape[1] #取图片的列数，即宽度
print("图像的高度：",rows,"图像的宽度：",cols)
delta_x=int(input("请输入x方向偏移量："))     #行的变化量，即平移量
delta_y=int(input("请输入y方向偏移量："))     #列的变化取，即平移量
center_x=int(input("请输入旋转中心x：（请在图像的宽度范围内输入）："))
center_y=int(input("请输入旋转中心y：（请在图像的宽度范围内输入）："))
center=[center_x,center_y]    #设置图片中心
result=np.zeros((rows,cols,3),dtype=np.uint8) #创建一样大小的转换结果
beta=int(input("请输入旋转角度："))*math.pi/180
s=1/float(input("请输入相似比例："))     #相似比
transform=np.array([[s*math.cos(beta),-s*math.sin(beta),delta_x],
                    [s*math.sin(beta), s*math.cos(beta),delta_y],
                    [0,0,1]])  #构建转换矩阵
cv2.imshow('original_picture',img)#显示原图片
for i in range(rows):
    for j in range(cols):
        img_pos=np.array([i-center[0],j-center[1],1])       #记录结果位置
        [x, y, z] = np.dot(transform, img_pos)          #转换为原图位置坐标
        x = int(x)+center[0]                      #取整
        y = int(y)+center[1]                      #取整
        if x >= rows or y >= cols or x < 0 or y < 0: #如果出界
            result[i][j] = 255                       #该点为白色
        else:
            result[i][j] = img[x][y]             #不出界把原图位置对应值取来
cv2.imshow('result_process', result)             #显示结果
cv2.waitKey(0)                                  #按任意键继续
