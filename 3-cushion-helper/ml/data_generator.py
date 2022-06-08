'''
Alt + Click : 흰색 공 배치
Shift + Click : 빨간색 공 배치
Ctrl + Click : 노란색 공 배치
Space : 공 무작위 배치
숫자키 (0,1,2,3,4) : 데이터 라벨링 + 저장
영어키 (p) : 미리 학습시킨 모델로 결과 예측
Esc : 종료
'''

import numpy as np
import random
import cv2
from keras.models import load_model

MODEL_NAME = "3-cushion-helper/temp_model.h5"

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 14

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 255)]
label_name = ["빨간공 왼쪽", "빨간공 오른쪽", "노란공 왼쪽", "노란공 오른쪽", "빈 쿠션"]
circles = [None, None, None]

model = load_model(MODEL_NAME)

class Circle:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color

def getRandomPos():
    return (random.randrange(BALL_RADIUS//4+1, WIDTH-BALL_RADIUS//4-1), random.randrange(BALL_RADIUS//4+1, HEIGHT-BALL_RADIUS//4-1))

def getTable(circles):
    table = np.zeros((HEIGHT*4, WIDTH*4, 3), dtype=np.float32)
    for circle in circles:
        if circle:
            cv2.circle(table, (circle.pos[0]*4, circle.pos[1]*4), BALL_RADIUS, circle.color, -1)
    return table

def writeData(y):
    with open('3-cushion-helper/ml/data.csv', 'a') as file:
        pos = sum([circle.pos for circle in circles] + [(y,)], ())
        data = ','.join(map(str, pos))
        file.write(data + '\n')
        print("Write Success:", data)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        if flags == 33: # White: Alt
            circles[0] = Circle((x//4, y//4), colors[0])
        elif flags == 17: # Red: Shift
            circles[1] = Circle((x//4, y//4), colors[1])
        elif flags == 9: # Yellow: Ctrl
            circles[2] = Circle((x//4, y//4), colors[2])
        cv2.imshow("table", getTable(circles))

cv2.imshow("table", getTable(circles))
cv2.setMouseCallback("table", onMouse)


while True:  
    key = cv2.waitKey()

    if ord('0') <= key <= ord('4'):
        writeData(chr(key))

    elif key == 32:
        for i in range(3):
            circles[i] = Circle(getRandomPos(), colors[i])
            cv2.imshow("table", getTable(circles))

    elif key == ord('p'):
        result = model.predict([sum([circle.pos for circle in circles], ())])[0]
        result = [(i, round(p*100)) for i, p in enumerate(result)]
        result.sort(key=lambda x: x[1], reverse=True)
        for label, prob in result:
            print(f"{prob}%: {label_name[label]}") 

    elif key == 27:
        break

cv2.destroyAllWindows()