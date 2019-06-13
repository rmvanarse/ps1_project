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
scale_x = SIGNATURE_CROP[3] - SIGNATURE_CROP[1]
scale_y = SIGNATURE_CROP[2] - SIGNATURE_CROP[0]

#Filtering
median_blurred_img = cv2.medianBlur(img, MEDIANBLUR_KERNEL_SIZE)
ret, bg_eliminated_img = cv2.threshold(median_blurred_img, 127, 255, cv2.THRESH_BINARY)

#temp_img = cv2.medianBlur(bg_eliminated_img, MEDIANBLUR_KERNEL_SIZE*3)
#emp_img = temp_img*4
img_contours, hierarchy = cv2.findContours(bg_eliminated_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#img_contours, hierarchy = cv2.findContours(bg_eliminated_img,1,2)

#cnt_temp = img_contours[0]
#M = cv2.moments(cnt_temp)

x2=0
y2=0
x1 = scale_x
y1 = scale_y
print(scale_x, scale_y, "DEBUG")
for cnt_temp in img_contours[:-1]:
	x,y,w,h = cv2.boundingRect(cnt_temp)
	x1 = min(x1, x)
	y1 = min(y1, y)
	x2 = max(x2, x+w)
	y2 = max(y2, y+h)
	#cv2.rectangle(bg_eliminated_img,(x,y),(x+w,y+h),(0,255,0))

print(x1, y1, x2, y2)
cv2.rectangle(bg_eliminated_img,(x1,y1),(x2,y2),(0,255,0))



#Display
cv2.imshow('image',bg_eliminated_img)
#cv2.imshow('im2', temp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()