import cv2
import numpy as np
import table_recognizer
import ball_recognizer

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 3

src = cv2.imread('3-cushion-helper/testimg/8.png')
table = table_recognizer.get_warped_table(src)

blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

yellow_ball_point = ball_recognizer.find_color_center(table, 'y')
white_ball_point = ball_recognizer.find_color_center(table, 'w')
red_ball_point = ball_recognizer.find_color_center(table, 'r')

cv2.circle(blank, yellow_ball_point, BALL_RADIUS, (0,255,255), -1)
cv2.circle(blank, white_ball_point, BALL_RADIUS, (255,255,255), -1)
cv2.circle(blank, red_ball_point, BALL_RADIUS, (0,0,255), -1)

cv2.imshow("result", blank)
cv2.waitKey()