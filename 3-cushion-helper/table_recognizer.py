import numpy as np
import cv2

WIDTH = 854
HEIGHT = 446

lower_blue = (100, 100, 100)
upper_blue = (120, 255, 255)


def find_corners(src):

    # 파란 색 검출
    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    table_img = cv2.inRange(src_hsv, lower_blue, upper_blue)

    # 빈공간 메꾸기(morphology closing)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    table_img_closing = cv2.morphologyEx(table_img, cv2.MORPH_CLOSE, k)

    # 가장 큰 파란 영역 컨투어 계산
    contours, _ = cv2.findContours(table_img_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    # 컨투어 근사
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        print('Table recognizaion failed!')
        exit()

    # 가로 point가 먼저 나타나도록 조정
    if np.linalg.norm(approx[1][0]-approx[0][0]) < np.linalg.norm(approx[2][0]-approx[1][0]):
        return (approx[1][0], approx[2][0], approx[3][0], approx[0][0])
    else:
        return (point[0] for point in approx)


def get_warped_table(src, points):
    
    # perspective transform matrix 계산
    src_point = np.array(points, dtype=np.float32)
    dst_point = np.array([[0, 0], [0, HEIGHT-1], [WIDTH-1, HEIGHT-1], [WIDTH-1, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_point, dst_point)

    # warp, 테이블 테두리 제거
    dst = cv2.warpPerspective(src, matrix, (WIDTH, HEIGHT))[20:-19, 20:-19]

    return dst

    src_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    yellow_img = cv2.inRange(src_hsv, lower_white, upper_white)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    yellow_img_opening = cv2.morphologyEx(yellow_img, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(yellow_img_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    mmt = cv2.moments(contour)
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])

cv2.drawContours(dst, contours, -1, (0, 255, 255), 2)
cv2.circle(dst, (cx, cy), 2, (0, 0, 255), -1)

cv2.imshow('dst', dst)
cv2.imshow('white', yellow_img_opening)

cv2.waitKey()
cv2.destroyAllWindows()