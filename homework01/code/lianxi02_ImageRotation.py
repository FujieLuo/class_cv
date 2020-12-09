#encoding:UTF-8
from PIL import Image
import math
import matplotlib.pyplot as plt

def rotate(image,degree,crop=False):
    im = Image.open(image)
    radius = math.pi * degree / 180
    width, height = im.size
    if not crop:
        X1 = math.ceil(abs(0.5 * height * math.cos(radius) + 0.5 * width * math.sin(radius)))
        X2 = math.ceil(abs(0.5 * height * math.cos(radius) - 0.5 * width * math.sin(radius)))
        Y1 = math.ceil(abs(-0.5 * height * math.sin(radius) + 0.5 * width * math.cos(radius)))
        Y2 = math.ceil(abs(-0.5 * height * math.sin(radius) - 0.5 * width * math.cos(radius)))
        H = int(2 * max(Y1, Y2))
        W = int(2 * max(X1, X2))
        dstwidth = W + 1
        dstheight = H + 1
    if crop:
        dstheight = height
        dstwidth = width
    im_new = Image.new('RGB', (dstwidth, dstheight), (255, 255, 255))
    for i in range(dstwidth):
        for j in range(dstheight):
            new_i = int(
                (i - 0.5 * dstwidth) * math.cos(radius) - (j - 0.5 * dstheight) * math.sin(radius) + 0.5 * width)
            new_j = int(
                (i - 0.5 * dstwidth) * math.sin(radius) + (j - 0.5 * dstheight) * math.cos(radius) + 0.5 * height)
            if new_i >= 0 and new_i < width and new_j >= 0 and new_j < height:
                im_new.putpixel((i, j), im.getpixel((new_i, new_j)))
    # im_new.show()
    sub = plt.subplot(1, 2, 1)
    sub.set_title("Src Img")
    plt.imshow(im)
    sub = plt.subplot(1, 2, 2)
    sub.set_title("Dst->Src & Nearest")
    plt.imshow(im_new)
    plt.show()

rotate('picture.jpg',30,crop=True)

