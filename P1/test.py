import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATCH_SIZE = 30
COLOR = ('b','g','r')
title = 'Project 1'

img1 = cv2.imread(os.path.join(os.getcwd(),'P1/1st.jpg'))
img2 = cv2.imread(os.path.join(os.getcwd(),'P1/2nd.jpg'))

img1 = cv2.resize(img1, (600, 800))
img2 = cv2.resize(img2, (600, 800))

def showHistogram(arr):
    for i,col in enumerate(COLOR):
        histr = cv2.calcHist([arr],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
    
def diff2PatchArray(patchArr1, patchArr2):
    hists1 = []
    hists2 = []
    for patch in patchArr1:
        # hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
        hist = cv2.calcHist([patch],[0,1],None,[256],[0,256])
        
        hists1.append(hist)
        
    for patch in patchArr2:
        # hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
        hist = cv2.calcHist([patch],[0,1],None,[256],[0,256])
        
        hists2.append(hist)
        
    compareResult = []
    
    

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(img1, (x-PATCH_SIZE//2, y-PATCH_SIZE//2), (x+PATCH_SIZE//2, y+PATCH_SIZE//2), (0, 0, 255), 2)
        cv2.imshow(title, img1)

cv2.imshow(title, img1)

cv2.setMouseCallback(title, onMouse)

cv2.waitKey()
cv2.destroyAllWindows()