import cv2
img=cv2.imread('picture.jpg')
cv2.imshow("original_picture",img)
level=3 #设置金字塔的层数
temp=img.copy()
gaosi_img=[]
for i in range(level):
    dst=cv2.pyrDown(temp)
    gaosi_img.append(dst)
    cv2.imshow("gsd"+str(i),dst)
    temp=dst.copy()
for i in range(level):
    dst=cv2.pyrUp(temp)
    gaosi_img.append(dst)
    cv2.imshow("gsu"+str(i),dst)
    temp=dst.copy()
cv2.waitKey(0)
cv2.destroyAllWindows()
