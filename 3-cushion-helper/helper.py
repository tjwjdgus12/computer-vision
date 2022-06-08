import cv2
import numpy as np
import table_recognizer
import ball_recognizer

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 3

src = cv2.imread('3-cushion-helper/test_img/4.jpg')
src = cv2.resize(src, (0, 0), fx=0.15, fy=0.15)
table = table_recognizer.get_warped_table(src)

blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

white_ball_point = ball_recognizer.find_color_center(table, 'w')
red_ball_point = ball_recognizer.find_color_center(table, 'r')
yellow_ball_point = ball_recognizer.find_color_center(table, 'y')

cv2.circle(blank, white_ball_point, BALL_RADIUS, (255,255,255), -1)
cv2.circle(blank, red_ball_point, BALL_RADIUS, (0,0,255), -1)
cv2.circle(blank, yellow_ball_point, BALL_RADIUS, (0,255,255), -1)

################################################################################

label_name = ["빨간공 왼쪽", "빨간공 오른쪽", "노란공 왼쪽", "노란공 오른쪽", "빈 쿠션"]

from keras.models import load_model
model = load_model("3-cushion-helper/temp_model.h5")

result = model.predict([white_ball_point + red_ball_point + yellow_ball_point])[0]
result = [(i, round(p*100)) for i, p in enumerate(result)]
result.sort(key=lambda x: x[1], reverse=True)
for label, prob in result:
    print(f"{prob}%: {label_name[label]}") 

cv2.imshow("src", src)
cv2.imshow("result", blank)
cv2.waitKey()

cv2.destroyAllWindows()