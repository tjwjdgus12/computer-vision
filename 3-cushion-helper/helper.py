import cv2
import table_recognizer
import ball_recognizer

src = cv2.imread('3-cushion-helper/testimg/4.jpg')
table = table_recognizer.get_warped_table(src)
ball_point = ball_recognizer.find_color_center(table, 'r')
print(ball_point)