import numpy as np
import cv2

WIDTH = 816 + 38
HEIGHT = 408 + 38

lower_red = (160, 100, 70)
upper_red = (190, 255, 255)

lower_white = (0, 0, 100)
upper_white = (360, 100, 255)

lower_yellow = (20, 100, 100)
upper_yellow = (40, 255, 255)

def find_color_center(src, color):
    if color == "r":
        lower_color, upper_color = lower_red, upper_red
    elif color == "y":
        lower_color, upper_color = lower_yellow, upper_yellow
    elif color == "w":
        lower_color, upper_color = lower_white, upper_white
    else:
        print("not supported color")
        return

    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(src_hsv, lower_color, upper_color)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    mmt = cv2.moments(contour)
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])

    return (cx, cy)