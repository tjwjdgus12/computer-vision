import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATCH_SIZE = 9

COLOR = ('b','g','r')
title = 'Project 1'

img1 = cv2.imread(os.path.join(os.getcwd(),'P1/1st.jpg'), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(os.getcwd(),'P1/2nd.jpg'), cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1, (300, 400))
img2 = cv2.resize(img2, (300, 400))

roi1, roi2 = [], []
roi_pos1, roi_pos2 = [], []

patchHistogram = []

def displayPatchHistogram(rowcount=2, colcount=4):
    fig, axes = plt.subplots(nrows=rowcount, ncols=colcount)
    
    flattenArr = axes.flatten()
    print(flattenArr)
    
    for i in range(0,rowcount):
        for j in range(0,colcount):
            
            axes[i][j].bar(np.arange(32), patchHistogram[i*colcount+j].ravel(), color=colors[i*colcount+j])
            axes[i][j].set_title('Picture '+str(i)+ '/ Patch '+str(j))
            
    fig.tight_layout()
    plt.show()
    
# displayPatchHistogram()
    
def getHistogram(arr):
    histr = cv2.calcHist([arr],[0],None,[32],[0,256])
    
    return histr
    
    # plt.bar(np.arange(32),histr.ravel())
    # plt.show()
    
def diff2PatchArray(patchArr1, patchArr2):
    hists1 = []
    hists2 = []
    
    
    for patch in patchArr1:
        # hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
        patchHistogram.append( getHistogram(patch) )
        hist = cv2.calcHist([patch],[0],None,[256],[0,256])
        
        hists1.append(hist)
        
    for patch in patchArr2:
        # hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
        patchHistogram.append( getHistogram(patch) )
        
        hist = cv2.calcHist([patch],[0],None,[256],[0,256])
        
        hists2.append(hist)
        
    compareResult = []
    
    for hist in hists1:
        localResult = []
        for hist2 in hists2:
            localResult.append( cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL) )
            
        compareResult.append(localResult)
        
    return compareResult
    
    

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, x2 = x - PATCH_SIZE // 2, x + PATCH_SIZE // 2
        y1, y2 = y - PATCH_SIZE // 2, y + PATCH_SIZE // 2
        param[1].append(param[0][y1:y2+1, x1:x2+1].copy())
        param[2].append((x, y))
        
        
        cv2.rectangle(param[0], (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(param[0], str(len(param[1])), (x-5, y-10), \
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255))
        cv2.imshow(title, param[0])

cv2.imshow(title, img1)
cv2.setMouseCallback(title, onMouse, [img1, roi1, roi_pos1])

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow(title, img2)
cv2.setMouseCallback(title, onMouse, [img2, roi2, roi_pos2])

cv2.waitKey()
cv2.destroyAllWindows()

result = diff2PatchArray(roi1, roi2)
displayPatchHistogram(2, len(roi1))

img = np.hstack((img1, img2))
pair = np.argmax(result, axis=1)
for i, j in enumerate(pair):
    cv2.line(img, roi_pos1[i], (roi_pos2[j][0] + 300, roi_pos2[j][1]), (0, 0, 255))

cv2.imshow(title, img)

cv2.waitKey()
cv2.destroyAllWindows()