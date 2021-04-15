# real-time computer vision, image library
import cv2
from PIL import Image, ImageFilter
import sys
import numpy as np

#read image
img = cv2.imread('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/Output.jpg')
img = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)
cv2.imwrite("output_ASL.jpg",img)
# img1 = cv2.imread("D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/Output.jpg")
# cv2.imwrite("Output.jpg",img1)

# for x in range(0,28):
#     for y in range(0,28):
#         Diff = np.subtract(img[x][y],img1[x][y]) 
#         print(Diff)
