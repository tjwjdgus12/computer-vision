import numpy as np
import cv2

WIDTH = 272
HEIGHT = 136
BALL_RADIUS = 14

def str2img(string):
    string = string.replace(',', ' ')
    ball_pos = list(map(int, string.split()))

    white_ball_pos = (ball_pos[0]*4, ball_pos[1]*4)
    red_ball_pos = (ball_pos[2]*4, ball_pos[3]*4)
    yellow_ball_pos = (ball_pos[4]*4, ball_pos[5]*4)

    table = np.zeros((HEIGHT*4, WIDTH*4, 3), dtype=np.float32)

    cv2.circle(table, white_ball_pos, 12, (255,255,255), -1)
    cv2.circle(table, red_ball_pos, BALL_RADIUS, (0,0,255), -1)
    cv2.circle(table, yellow_ball_pos, BALL_RADIUS, (0,255,255), -1)

    return table


if __name__ == "__main__":

    while True:
        ball_pos_str = input()
        if ball_pos_str == "": continue

        table_img = str2img(ball_pos_str)
        
        cv2.imshow("result", table_img)
        cv2.waitKey()
        cv2.destroyAllWindows()