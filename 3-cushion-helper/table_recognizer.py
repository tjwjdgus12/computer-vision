import numpy as np
import cv2

src = cv2.imread('3-cushion-helper/testimg1.png')

if src is None:
    print('Image load failed!')
    exit()

# 파란 색 검출 - 색상 값 범위 조절
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
lower_blue = (100, 150, 150)
upper_blue = (120, 255, 255)
table_img = cv2.inRange(src_hsv, lower_blue, upper_blue)

# 빈공간 메꾸기(morphology closing) - 커널 모양 및 크기 조절
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
table_img_closing = cv2.morphologyEx(table_img, cv2.MORPH_CLOSE, k)

# 가장 큰 파란 영역 컨투어 계산
contours, _ = cv2.findContours(table_img_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=lambda x: cv2.contourArea(x))

# 컨투어 근사 - epsilon 상수 조절
epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)


cv2.drawContours(src, [approx], -1, (255,0,0), 2)
for point in approx:
  cv2.circle(src, point[0], 3, (0,255,0), -1)

cv2.imshow('table corner', src)
cv2.waitKey()

cv2.destroyAllWindows()