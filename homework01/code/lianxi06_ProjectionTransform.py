import cv2
import numpy as np
img = cv2.imread('picture.jpg')#读取图片成(400,400,3)的张量，左上为原点
rows = img.shape[0] #取图片的行数，即高度
cols = img.shape[1] #取图片的列数，即宽度
print("图像的高度：",rows,"图像的宽度：",cols)
center=[0,0]    #设置图片中心
result=np.zeros((rows,cols,3),dtype=np.uint8) #创建一样大小的转换结果
h11=float(input("请输入射影矩阵h11："))
h12=float(input("请输入射影矩阵h12："))
h13=float(input("请输入射影矩阵h13："))
h21=float(input("请输入射影矩阵h21："))
h22=float(input("请输入射影矩阵h22："))
h23=float(input("请输入射影矩阵h23："))
h31=float(input("请输入射影矩阵h31："))
h32=float(input("请输入射影矩阵h32："))
transform=np.array([[h11,h12,h13],
                    [h21, h22,h23],
                    [h31,h32,1]])   #转换矩阵
transform=np.linalg.inv(transform)
cv2.imshow('original_picture',img)#显示原图片
Z=1
for i in range(rows):
    for j in range(cols):
        img_pos=np.array([i-center[0],j-center[1],1])           #记录结果位置
        [x, y, z] = np.dot(transform, img_pos)          #转换为原图位置坐标
        x = int(x/Z)+center[0]                      #取整
        y = int(y/Z)+center[1]                      #取整
        if x >= rows or y >= cols or x < 0 or y < 0: #如果出界
            result[i][j] = 255                       #该点为白色
        else:
            result[i][j] = img[x][y]            #不出界把原图位置对应值取来
cv2.imshow('result_process', result)            #显示结果
cv2.waitKey(0)                                  #按任意键继续
