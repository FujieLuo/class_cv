import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

#数据初始处理
src = cv.imread('picture.jpg')
#plt.imshow(src[:,:,[2,1,0]])
#plt.show()
#准备图片成灰度图
im1=cv.cvtColor(src,cv.COLOR_BGR2GRAY).astype(float)
#plt.imshow(im1,cmap='gray')
#plt.show()
#定义函数
def create_gaussion_kernel(sigma1=1.0,sigma2=1.0): #高斯核滤波算子
    sum=0.0
    gaussian=np.zeros([5,5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * ((i - 2)**2 / (sigma1)**2+((j - 2)**2 / (sigma2)**2))) / (2 * math.pi * sigma1 * sigma2)
            sum=sum+gaussian[i,j]

    gaussian=gaussian/sum
    return gaussian

def gauss_5x5(a):#高斯滤波算子计算函数
    computer=np.ones(dtype=float,shape=(5,5))
    for i in range(-2,3):
        for j in  range(-2,3):
            computer[i+2,j+2]=pow(math.e,(-i*i-j*j)/(2*a*a))
    computer=computer/np.min(computer)
    computer=computer.astype(int)+1
    computer=np.where(computer==2,1,computer)
    computer[2,2]=56
    computer=computer.astype(float)
    return computer

def convolution(computer,image_1):#卷积函数
    image_2=image_1.copy()
    size_i=int(computer.shape[0]/2)
    a=[0,0]
    for b in range(0,2):
        if computer.shape[b]%2==0:
            a[b]=0
        else:
            a[b]=1
    size_j=int(computer.shape[1]/2)
    sum_1=np.sum(computer)
    if sum_1==0:
        sum_1=1
    for i_1 in range(size_i,image_1.shape[0]-size_i):
        for j_1 in range(size_j,image_1.shape[1]-size_j):
            image_2[i_1,j_1]=np.sum(image_1[i_1-size_i:i_1+size_i+a[1],j_1-size_j:j_1+size_j+a[0]]*computer)/sum_1
    return image_2

def edg(edg_cp_x,edg_cp_y,im): #求梯度模
    im_1=im.copy()
    im_a=convolution(edg_cp_x,im_1)
    im_b=convolution(edg_cp_y,im_1)
    im_2=np.sqrt(im_a*im_a+im_b*im_b)
    return im_2

def point_extract(image,a,b=255):#返回角点检测地址
    image=image*255/np.max(image)     #灰度范围从min-max转到0-255
    # image=np.absolute(image)
    arry=np.where(image>=a)
    arry=np.array(arry)
    return arry

#%%进行滤波
#print(gauss_5x5(1))
detect=create_gaussion_kernel(1.0,1.0)
#print(detect)
im2=convolution(detect,im1)


#%%roberts边缘算子计算2x2
edg1_cp_x=np.array([1,0,0,-1]).reshape(2,2)
edg1_cp_y=np.array([0,-1,1,0]).reshape(2,2)
im5=edg(edg1_cp_x,edg1_cp_y,im2)
#%%sobel边缘算子计算3x3
edg2_cp_x=np.array([-1,0,1,-1,0,1,-1,0,1]).reshape(3,3)
edg2_cp_y=-edg2_cp_x.T
im6=edg(edg2_cp_x,edg2_cp_y,im2)
#%% Harris角点检测
H_x=np.array([-1,0,1,-1,0,1,-1,0,1],dtype=float).reshape(3,3)
H_y=-H_x.T
H_i_x=convolution(H_x,im2)   #x方向梯度
H_i_y=convolution(H_y,im2)   #y方向梯度
H_A=convolution(detect,H_i_x*H_i_x)     #w（x,y）*Ix^2
H_B=convolution(detect,H_i_y*H_i_y)     #w（x,y）*Iy^2
H_C=convolution(detect,H_i_x*H_i_y)     #w（x,y）*Ix*Iy
M=np.array([[H_A,H_C],
            [H_C,H_B]])
M_T=M.transpose((2,3,0,1))  #调换索引值
M_det=np.linalg.det(M_T)    #求|M| 行列式的值
M_trace=np.trace(M)     #求M的迹
M_trace=M_trace.astype(float)   #转换成float
im3=M_det-0.04*M_trace*M_trace      #角点相应函数
#%%归一化角点提取
l1=point_extract(im3,110)             #提取相应函数大于110的地址 给l1

im7=np.float32(im1)                    #转换float32
im4=cv.cornerHarris(im7,2,3,0.04)       #im7未高斯模糊，其输入图像必须是float32 最后一个参数是相应函数的elpha，blockSize，表示邻域的大小=2；ksize，表示Sobel()算子的孔径大小=3
l2=point_extract(im4,160)             #提取相应函数大于160的地址 给l2


def harris(img,sigma1,sigma2,a,r):
    detect=create_gaussion_kernel(sigma1,sigma2)
    im2=convolution(detect,img)
    H_x=np.array([-1,0,1,-1,0,1,-1,0,1],dtype=float).reshape(3,3)
    H_y=-H_x.T
    H_i_x=convolution(H_x,im2)   #x方向梯度
    H_i_y=convolution(H_y,im2)   #y方向梯度
    H_A=convolution(detect,H_i_x*H_i_x)     #w（x,y）*Ix^2
    H_B=convolution(detect,H_i_y*H_i_y)     #w（x,y）*Iy^2
    H_C=convolution(detect,H_i_x*H_i_y)     #w（x,y）*Ix*Iy
    M=np.array([[H_A,H_C],
                [H_C,H_B]])
    M_T=M.transpose((2,3,0,1))  #调换索引值
    M_det=np.linalg.det(M_T)    #求|M| 行列式的值
    M_trace=np.trace(M)     #求M的迹
    M_trace=M_trace.astype(float)   #转换成float
    im3=M_det-a*M_trace*M_trace      #角点相应函数
    #%%归一化角点提取
    l1=point_extract(im3,r)             #提取相应函数大于110的地址 给l1
    plt.figure( figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.scatter(l1[1, :], l1[0, :], marker='o', color='red', s=5)
    plt.show()


# src = cv.imread('picture.jpg')
# im10=cv.cvtColor(src,cv.COLOR_BGR2GRAY).astype(float)
# src2 = cv.imread('picture.jpg')
# im11=(cv.cvtColor(src2,cv.COLOR_BGR2GRAY).astype(float)+20)

src = cv.imread('big1.JPG')
im10=cv.cvtColor(src,cv.COLOR_BGR2GRAY).astype(float)
src2 = cv.imread('big2.JPG')
im11=(cv.cvtColor(src2,cv.COLOR_BGR2GRAY).astype(float))

# src3 = cv.imread('picture6.jpg')
# im12=cv.cvtColor(src3,cv.COLOR_BGR2GRAY).astype(float)
# harris(im10,0.1,0.1,0.1,110)
# harris(im10,0.5,0.5,0.1,110)
# harris(im10,1.0,1.0,0.1,110)
# harris(im10,2.0,2.0,0.1,110)
# harris(im10,5.0,5.0,0.1,110)
# harris(im10,10.0,10.0,0.1,110)

# harris(im10,5.0,5.0,0.21,110)
# harris(im10,5.0,5.0,0.22,110)
# harris(im10,5.0,5.0,0.25,110)

harris(im10,5,5,0.05,110)
harris(im11,5,5,0.05,110)
#harris(im12,5,5,0.05,110)



harris(im10,5,5,0.1,110)
harris(im10,5,5,0.2,110)
harris(im10,5.0,5.0,0.5,110)
harris(im10,5.0,5.0,0.8,110)



#src = cv.imread('picture.jpg')
# im11=(cv.cvtColor(src,cv.COLOR_BGR2GRAY).astype(float)+10)
# harris(im11,1.0,1.0,0.04,110)
