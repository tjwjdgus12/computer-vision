import cv2
import numpy as np

PATCH_SIZE = 30

title = 'Project 1'

img1 = cv2.imread('P1/1st.jpg')
img2 = cv2.imread('P1/2nd.jpg')

img1 = cv2.resize(img1, (600, 800))
img2 = cv2.resize(img2, (600, 800))

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(img1, (x-PATCH_SIZE//2, y-PATCH_SIZE//2), (x+PATCH_SIZE//2, y+PATCH_SIZE//2), (0, 0, 255), 2)
        cv2.imshow(title, img1)

cv2.imshow(title, img1)

cv2.setMouseCallback(title, onMouse)

cv2.waitKey()
cv2.destroyAllWindows()