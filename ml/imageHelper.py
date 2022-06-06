# import tensorflow as tf
# from tensorflow import keras

import numpy as np
import random
import cv2

import itertools as it
# import matplotlib.pyplot as plt

from PIL import Image

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 3

# data = open('data.csv','a')

def getRandomX():
    return random.randrange(BALL_RADIUS+1,HEIGHT-BALL_RADIUS-1)

def getRandomY():
    return random.randrange(BALL_RADIUS+1,WIDTH-BALL_RADIUS-1)

i=0
while True:
    im = Image.new("RGB",(WIDTH*4,HEIGHT*4),(0,0,0))

    img = np.asarray(im)

    p1 = (getRandomY(), getRandomX())
    p2 = (getRandomY(), getRandomX())
    p3 = (getRandomY(), getRandomX())

    tmp = "14	48	208	98	57	130"
    nlist = list(map(int, tmp.split()))

    p1 = (nlist[0], nlist[1])
    p2 = (nlist[2], nlist[3])
    p3 = (nlist[4], nlist[5])
    
    p1_ = (p1[0]*4, p1[1]*4)
    p2_ = (p2[0]*4, p2[1]*4)
    p3_ = (p3[0]*4, p3[1]*4)

    cv2.circle(img,p1_,BALL_RADIUS*4,(255,255,255),-1)
    cv2.circle(img,p2_,BALL_RADIUS*4,(0,0,255),-1)
    cv2.circle(img,p3_,BALL_RADIUS*4,(0,255,255),-1)


    cv2.imshow('test'+str(i),img)
    cv2.waitKey(1)

    num = int(input())

    print(num)


    cv2.destroyWindow("test"+str(i))

    if num == -1:
        continue

    data.writelines(str(p1[0])+","+str(p1[1])+","+str(p2[0])+","+str(p2[1])+","+str(p3[0])+","+str(p3[1])+","+str(num)+"\n")
    data.flush()

    