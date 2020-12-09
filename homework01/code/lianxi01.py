import cv2
import numpy as np
img = cv2.imread('picture.jpg')#读取图片成(400,400,3)的张量，左上为原点
rows = img.shape[0] #取图片的行数，即高度
cols = img.shape[1] #取图片的列数，即宽度
print(rows,cols)
location=np.zeros((rows,cols,2),dtype=np.uint8) #创建一样大小的转换结果
print(location.shape)


# delta_x=int(input("请输入x方向偏移量："))     #行的变化量，即平移量
# delta_y=int(input("请输入y方向偏移量："))     #列的变化取，即平移量
delta_x=20    #行的变化量，即平移量
delta_y=20   #列的变化取，即平移量

transform=np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])   #转换矩阵
cv2.imshow('original_picture',img)#显示原图片
for i in range(rows):
    for j in range(cols):
        img_pos=np.array([i,j,1])           #原始的像素点的坐标位置矩阵
        [x, y, z] = np.dot(transform, img_pos) #转换后的位置坐标矩阵
        x = int(x)                          #取整
        y = int(y)                          #取整
        location[i][j]=x,y
new_rows=rows+delta_y
new_cols= cols+delta_x
result = np.zeros((new_rows,new_cols, 3), dtype=np.uint8)  # 创建一样大小的转换结果
# for ii in range(new_rows):
#     for jj in range(new_cols):
for i in range(rows):
    for j in range(cols):
        result[location[i][j][0]][location[i][j][1]]=img[i][j]
cv2.imshow('result_process', result)            #显示结果
cv2.waitKey(0)                                  #按任意键继续
print(location)
print(location[0][1][0],location[0][1][1])
