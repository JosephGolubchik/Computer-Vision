import time
import cv2
import keyboard
import numpy as np

def nothing(x):
    pass

if __name__ == '__main__':
    cv2.namedWindow("frame")
    cv2.createTrackbar("high", "frame", 0, 100, nothing)
    cv2.createTrackbar("low", "frame", 0, 100, nothing)

    img = cv2.imread('img/lane1.jpeg')
    threshold_low = 0
    threshold_high = 255
    while(1):
        edges = cv2.Canny(img, threshold_low, threshold_high)
        cv2.imshow("frame", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if keyboard.is_pressed('a'):
            threshold_high -= 10
            print(threshold_low, threshold_high)
        if keyboard.is_pressed('s'):
            threshold_high += 10
            print(threshold_low, threshold_high)
        if keyboard.is_pressed('z'):
            threshold_low -= 10
            print(threshold_low, threshold_high)
        if keyboard.is_pressed('x'):
            threshold_low += 10
            print(threshold_low, threshold_high)
    cv2.destroyAllWindows()