import cv2
import numpy as np

title = 'Project 1' 

img1 = cv2.imread('P1/1st.jpg')
img2 = cv2.imread('P1/2nd.jpg')

img1 = cv2.resize(img1, (600, 800))
img2 = cv2.resize(img2, (600, 800))

def onMouse(event, x, y, flags, param):
    print(event, x, y, )
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(img1, (x-10, y-10), (x+10, y+10), (255, 255, 255), -1)
        cv2.imshow(title, img1)

cv2.imshow(title, img1)

cv2.setMouseCallback(title, onMouse)

cv2.waitKey()
cv2.destroyAllWindows()