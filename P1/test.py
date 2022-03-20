import cv2
import numpy as np

PATCH_SIZE = 9

title = 'Project 1'

img1 = cv2.imread('P1/1st.jpg')
img2 = cv2.imread('P1/2nd.jpg')

img1 = cv2.resize(img1, (600, 800))
img2 = cv2.resize(img2, (600, 800))

roi1 = []
roi2 = []

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, x2 = x - PATCH_SIZE // 2, x + PATCH_SIZE // 2 + 1
        y1, y2 = y - PATCH_SIZE // 2, y + PATCH_SIZE // 2 + 1
        cv2.rectangle(param[0], (x1, y1), (x2, y2), (0, 0, 255), 1)
        param[1].append(param[0][y1:y2, x1:x2])
        cv2.imshow(title, param[0])

cv2.imshow(title, img1)

cv2.setMouseCallback(title, onMouse, [img1, roi1])

cv2.waitKey()
cv2.destroyAllWindows()