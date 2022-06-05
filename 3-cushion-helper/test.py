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
table_img = cv2.inRange(src_hsv, lower_blue, upper_blue)

cv2.imshow('src', src)
cv2.imshow('dst', table_img)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
table_img_closing = cv2.morphologyEx(table_img, cv2.MORPH_CLOSE, k)

cv2.imshow('morpholgy', table_img_closing)

contours, _ = cv2.findContours(table_img_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour = max(contours, key=lambda x: cv2.contourArea(x))

epsilon = 0.04 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
cv2.drawContours(src, [approx], -1, (255,0,0), -1)

for point in approx:
    cv2.circle(src, point[0], 3, (0,0,255), -1)

# some_point = contour[0][0]
# corner1 = max(contour, key=lambda x: np.linalg.norm(x[0]-some_point))[0]
# corner2 = max(contour, key=lambda x: np.linalg.norm(x[0]-corner1))[0]
# corner3 = max(contour, key=lambda x: np.linalg.norm(x[0]-corner2)+np.linalg.norm(x[0]-corner1))[0]
# corner4 = max(contour, key=lambda x: cv2.contourArea([[point] for point in [corner1, corner2, corner3]] + x))[0]

# cv2.circle(src, corner1, 3, (0,0,255), -1)
# cv2.circle(src, corner2, 3, (0,0,255), -1)
# cv2.circle(src, corner3, 3, (0,0,255), -1)
# cv2.circle(src, corner4, 3, (255,0,255), -1)


cv2.imshow('contour', src)
cv2.waitKey()

cv2.destroyAllWindows()