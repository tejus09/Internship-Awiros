import cv2
import matplotlib.pyplot as plt
import numpy as np
import face_recognition as fr
from PIL import Image
import time
def pixel(img, center, x , y):
    try:
        if img[x][y] >= center:
            return 1
    except:
        pass
    return 0
def lbp(img, x, y):
    center = img[x][y]
    li = []
    li.extend([pixel(img, center, x-1, y-1), pixel(img, center, x-1, y), pixel(img, center, x-1, y+1), pixel(img, center, x, y+1), pixel(img, center, x+1, y+1), pixel(img, center, x+1, y), pixel(img, center, x+1, y-1), pixel(img, center, x, y-1)])
    val = 0
    power = [1,2,4,8,16,32,64,128,256,512,1024]
    for i in range(len(li)):
        val += li[i]*power[i]
    return val

path1 = "C:/Users/tejus/Downloads/e1.jpg"
path2 = "C:/Users/tejus/Downloads/elon.jpg"

img1 = cv2.imread(path1, 1)
img2 = cv2.imread(path2, 1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

h1, w1, c1 = img1.shape
h2, w2, c2 = img2.shape
img_lbp1 = np.zeros((h1, w1), dtype = int)
img_lbp2 = np.zeros((h1, w1), dtype = int)

for i in range(0, h1):
    for j in range(0, w1):
        img_lbp1[i, j] = lbp(gray1, i, j)
        img_lbp2[i, j] = lbp(gray2, i, j)

plt.imshow(img_lbp2, cmap='gray')
plt.show()
print(i)