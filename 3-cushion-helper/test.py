import os
import numpy as np
import cv2

src = cv2.imread('3-cushion-helper/testimg1.png')

if src is None:
    print('Image load failed!')
    exit()

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

lower_blue = (100, 150, 150)
upper_blue = (120, 255, 255)
dst = cv2.inRange(src_hsv, lower_blue, upper_blue)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

imgray = dst
contours, _ = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(src, [approx], -1, (0,255,0), 3)

cv2.imshow('contour', src)
cv2.waitKey()

cv2.destroyAllWindows()