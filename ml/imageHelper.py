# import tensorflow as tf
# from tensorflow import keras

import numpy as np
import random
import cv2

import itertools as it
# import matplotlib.pyplot as plt

from PIL import Image

WIDTH = 816
HEIGHT = 408
BALL_RADIUS = 10

data = open('data.csv','a')

def getRandomX():
    return random.randrange(BALL_RADIUS+1,HEIGHT-BALL_RADIUS-1)

def getRandomY():
    return random.randrange(BALL_RADIUS+1,WIDTH-BALL_RADIUS-1)

i=0
while True:
    im = Image.new("RGB",(WIDTH,HEIGHT),(0,0,0))

    img = np.asarray(im)

    p1 = (getRandomY(), getRandomX())
    p2 = (getRandomY(), getRandomX())
    p3 = (getRandomY(), getRandomX())


    cv2.circle(img,p1,BALL_RADIUS,(255,255,255),-1)
    cv2.circle(img,p2,BALL_RADIUS,(0,0,255),-1)
    cv2.circle(img,p3,BALL_RADIUS,(0,255,255),-1)


    cv2.imshow('test'+str(i),img)
    cv2.waitKey(1)

    num = int(input())

    print(num)


    cv2.destroyWindow("test"+str(i))

    if num == -1:
        continue

    data.writelines(str(p1[0])+","+str(p1[1])+","+str(p2[0])+","+str(p2[1])+","+str(p3[0])+","+str(p3[1])+","+str(num)+"\n")
    data.flush()

    