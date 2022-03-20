import cv2, numpy as np
import matplotlib.pylab as plt

img1 = cv2.imread('img/taekwonv1.jpg')
img2 = cv2.imread('img/taekwonv2.jpg')
img3 = cv2.imread('img/taekwonv3.jpg')
img4 = cv2.imread('img/dr_ochanomizu.jpg')

imgs = [img1, img2, img3, img4]
hists = []
for img in imgs :
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
    hists.append(hist)

origin = hists[0]

for i, (hist, img) in enumerate(zip(hists, imgs)):
    ret = cv2.compareHist(origin, hist, cv2.HISTCMP_CORREL)
    print("img%d:%7.2f"% (i+1 , ret), end='\t')

print()

cv2.imshow('img', np.hstack((img1, img2, img3, img4)))
cv2.waitKey()
cv2.destroyAllWindows()