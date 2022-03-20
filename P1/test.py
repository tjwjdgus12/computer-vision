import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATCH_SIZE = 9

COLOR = ('b','g','r')
title = 'Project 1'

img1 = cv2.imread(os.path.join(os.getcwd(),'P1/1st.jpg'))
img2 = cv2.imread(os.path.join(os.getcwd(),'P1/2nd.jpg'))

img1 = cv2.resize(img1, (600, 800))
img2 = cv2.resize(img2, (600, 800))

roi1 = []
roi2 = []

def showHistogram(arr):
    for i,col in enumerate(COLOR):
        histr = cv2.calcHist([arr],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

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

cv2.imshow(title, img2)
cv2.setMouseCallback(title, onMouse, [img2, roi2])

cv2.waitKey()
cv2.destroyAllWindows()