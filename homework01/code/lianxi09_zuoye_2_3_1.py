import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
#读取图片
img=cv.imread('picture.jpg')
#创建高斯矩阵核
def create_gaussion_kernel(sigma1=1.0,sigma2=1.0):
    sum=0.0
    gaussian=np.zeros([5,5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * ((i - 2)**2 / (sigma1)**2+((j - 2)**2 / (sigma2)**2))) / (2 * math.pi * sigma1 * sigma2)
            sum=sum+gaussian[i,j]

    gaussian=gaussian/sum
    return gaussian
def rgbTogray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

#1 高斯滤波
def gaussion_filter(img,sigma1,sigma2):
    gaussian=create_gaussion_kernel(sigma1, sigma2)
    gray=rgbTogray(img)
    plt.figure(figsize=(3,3))
    plt.imshow(gray, cmap="gray")
    plt.title('original')
    W, H = gray.shape
    new_gray = np.zeros([W - 4, H - 4])
    for i in range(W - 4):
        for j in range(H - 4):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波
    plt.figure(figsize=(3, 3))
    plt.imshow(new_gray, cmap="gray")
    plt.title('gaussion')
    return new_gray

#2 求一阶梯度幅值
def one_d(new_gray):
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值
    plt.figure(figsize=(3, 3))
    plt.imshow(d, cmap="gray")
    plt.title('Amplitude_Picture')
def one_drection(new_gray):
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.arctan((dy[i, j])/dx[i, j])  # 图像梯度方向作为图像强度值
    plt.figure(figsize=(3, 3))
    plt.imshow(d, cmap="gray")
    plt.title('Drection_Picture')
new_gray=gaussion_filter(img,1.0,1.0)
one_d(new_gray)
one_drection(new_gray)
new_gray=gaussion_filter(img,5.0,5.0)
one_d(new_gray)
one_drection(new_gray)
new_gray=gaussion_filter(img,0.5,0.5)
one_d(new_gray)
one_drection(new_gray)
plt.show()
