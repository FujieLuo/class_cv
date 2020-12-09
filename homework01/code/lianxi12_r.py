import os
from PIL import Image
img = Image.open('picture.jpg')
img1 = img.transpose(Image.ROTATE_30)  # 将图片旋转90度
#img = img.transpose(Image.ROTATE_180)  # 将图片旋转180度
#img = img.transpose(Image.ROTATE_270)  # 将图片旋转270度
img.show(img)
img1.show(img1)
img1.save("Img1.jpg")
