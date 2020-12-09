import cv2
import numpy as np
img = cv2.imread('picture.jpg')#读取图片成(400,400,3)的张量，左上为原点
rows = img.shape[0] #取图片的行数，即高度
cols = img.shape[1] #取图片的列数，即宽度
print("图像的高度：",rows,"图像的宽度：",cols)
center=[0,0]    #设置图片中心
result=np.zeros((rows,cols,3),dtype=np.uint8) #创建一样大小的转换结果
a11=float(input("请输入仿射矩阵s_x："))
a12=float(input("请输入仿射矩阵a_x："))
a21=float(input("请输入仿射矩阵a_y："))
a22=float(input("请输入仿射矩阵s_y："))
delta_x=int(input("请输入x方向偏移量t_x："))     #行的变化量，即平移量
delta_y=int(input("请输入y方向偏移量t_y："))     #列的变化取，即平移量
transform=np.array([[a11,a12,delta_x],
                    [a21, a22,delta_y],
                    [0,    0,     1]])   #转换矩阵
cv2.imshow('original_picture',img)#显示原图片
for i in range(rows):
    for j in range(cols):
        img_pos=np.array([i-center[0],j-center[1],1])           #记录结果位置
        [x, y, z] = np.dot(transform, img_pos)          #转换为原图位置坐标
        x = int(x)+center[0]                      #取整
        y = int(y)+center[1]                      #取整
        if x >= rows or y >= cols or x < 0 or y < 0: #如果出界
            result[i][j] = 255                       #该点为白色
        else:
            result[i][j] = img[x][y]            #不出界把原图位置对应值取来
cv2.imshow('result_process', result)            #显示结果
cv2.waitKey(0)                                  #按任意键继续
