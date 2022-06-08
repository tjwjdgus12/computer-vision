import numpy as np
import cv2

WIDTH = 272 + 6*2
HEIGHT = 136 + 6*2

lower_blue = (100, 100, 100)
upper_blue = (120, 255, 255)


def find_corners(src, debug=False):
    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    table_img = cv2.inRange(src_hsv, lower_blue, upper_blue)
    
    if debug: cv2.imshow("debug", src); cv2.waitKey()
    if debug: cv2.imshow("debug", table_img); cv2.waitKey()

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    table_img_closing = cv2.morphologyEx(table_img, cv2.MORPH_CLOSE, k)

    if debug: cv2.imshow("debug", table_img_closing); cv2.waitKey()

    contours, _ = cv2.findContours(table_img_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        print('Table recognizaion failed!')
        exit()

    if debug:
        src_tmp = src.copy()
        cv2.drawContours(src_tmp, [approx], -1, (0, 255, 0), 3)
        cv2.imshow("debug", src_tmp);
        cv2.waitKey()
    
    return approx


def get_warped_table(src, approx, debug=False):
    side_length = [np.linalg.norm(approx[i][0]-approx[i+1][0]) for i in range(-1, 3)]
    upper_left_point_idx = min(range(4), key=lambda i: approx[i][0][0]+approx[i][0][1])
    
    if side_length[upper_left_point_idx] > side_length[(upper_left_point_idx+1)%4]:
        si = upper_left_point_idx
    else:
        si = (upper_left_point_idx + 1) % 4

    src_point = np.array([approx[si][0], approx[(si+1)%4][0], approx[(si+2)%4][0], approx[(si+3)%4][0]], dtype=np.float32)
    dst_point = np.array([[0, 0], [0, HEIGHT-1], [WIDTH-1, HEIGHT-1], [WIDTH-1, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_point, dst_point)

    dst = cv2.warpPerspective(src, matrix, (WIDTH, HEIGHT))[5:-5, 7:-7]
    if debug: cv2.imshow("debug", dst); cv2.waitKey()
    return dst


if __name__ == "__main__":
    src = cv2.imread('3-cushion-helper/test_img/2.jpg')
    src = cv2.resize(src, (0,0), fx=0.15, fy=0.15)
    get_warped_table(src, debug=True)