"""

Pre-Processing 
Uses: CV2
Classical filters

"""

import cv2
import numpy as np

IMAGE_FILE = 'signature_sketch.jpg'

SIGNATURE_CROP = [0,0,210,460] #Temporary values
MEDIANBLUR_KERNEL_SIZE = 5

#Load Image

img = cv2.imread(IMAGE_FILE, 0)
print("Size: ", img.shape)
img = img[SIGNATURE_CROP[0]:SIGNATURE_CROP[2], SIGNATURE_CROP[1]:SIGNATURE_CROP[3]]


#Filtering
median_blurred_img = cv2.medianBlur(img, MEDIANBLUR_KERNEL_SIZE)
ret, bg_eliminated_img = cv2.threshold(median_blurred_img, 127, 255, cv2.THRESH_BINARY)


img_contours, hierarchy = cv2.findContours(bg_eliminated_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt_temp = img_contours[0]
M = cv2.moments(cnt_temp)

"""
x,y,w,h = cv2.boundingRect(cnt_temp)
print(x,y,w,h)
cv2.rectangle(bg_eliminated_img,(x,y),(x+w,y+h),(0,255,0))
"""

#Display
cv2.imshow('image',bg_eliminated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()