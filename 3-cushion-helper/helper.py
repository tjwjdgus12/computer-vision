import cv2
import numpy as np
import math
import table_recognizer
import ball_recognizer

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 3

src = cv2.imread('3-cushion-helper/test_img/4.jpg')
src = cv2.resize(src, (0, 0), fx=0.15, fy=0.15)
contour = table_recognizer.find_corners(src)
table = table_recognizer.get_warped_table(src, contour)

blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

white_ball_point, _ = ball_recognizer.find_color_center(table, 'w')
red_ball_point, _ = ball_recognizer.find_color_center(table, 'r')
yellow_ball_point, _ = ball_recognizer.find_color_center(table, 'y')

cv2.circle(blank, white_ball_point, BALL_RADIUS, (255,255,255), -1)
cv2.circle(blank, red_ball_point, BALL_RADIUS, (0,0,255), -1)
cv2.circle(blank, yellow_ball_point, BALL_RADIUS, (0,255,255), -1)

################################################################################

label_name = [
    "Hit the left side of red ball",
    "Hit the right side of red ball",
    "Hit the left side of yellow ball",
    "Hit the right side of yellow ball",
    "Do bank shot"
]

from keras.models import load_model
model = load_model("3-cushion-helper/temp_model.h5")

result = model.predict([white_ball_point + red_ball_point + yellow_ball_point])[0]
result = [(i, round(p*100)) for i, p in enumerate(result)]
result.sort(key=lambda x: x[1], reverse=True)
for label, prob in result:
    print(f"{prob}%: {label_name[label]}") 

mask = np.zeros_like(src)
cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
masked = cv2.bitwise_and(src, mask)

white_ball_point, _ = ball_recognizer.find_color_center(masked, 'w', correction=False)

if result[0][0] in [0, 1]:
    aim_ball_point, aim_ball_radius = ball_recognizer.find_color_center(masked, 'r', correction=False)
elif result[0][0] in [2, 3]:
    aim_ball_point, aim_ball_radius = ball_recognizer.find_color_center(masked, 'y', correction=False)
else:
    aim_ball_point = None

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

if aim_ball_point:
    distance = np.linalg.norm(np.array(white_ball_point) - np.array(aim_ball_point))
    sin_value = aim_ball_radius / distance
    radian = math.asin(sin_value) * 2
    if result[0][0] in [0, 2]:
        radian *= -1
    new_point = rotate(white_ball_point, aim_ball_point, radian)
    cv2.arrowedLine(src, white_ball_point, new_point, (0,0,0), 5)
    cv2.arrowedLine(src, white_ball_point, new_point, (255,255,255), 2)

cv2.putText(src, f"{label_name[result[0][0]]} ({result[0][1]}%)", (30, 50), 4, 0.6, (0,0,0), 3)
cv2.putText(src, f"{label_name[result[0][0]]} ({result[0][1]}%)", (30, 50), 4, 0.6, (255,255,255), 1)

cv2.imshow("src", src)
cv2.imshow("result", blank)
cv2.waitKey()

cv2.destroyAllWindows()