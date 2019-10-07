import numpy as np
import cv2


def AvgHSV(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg_h = np.average(hsv_img[:,:,0])
    avg_s = np.average(hsv_img[:, :, 1])
    avg_v = np.average(hsv_img[:, :, 2])
    return np.array([avg_h, avg_s, avg_v])

def AvgRGB(img):
    avg_r = np.average(img[:,:,0])
    avg_g = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    return np.array([avg_r, avg_g, avg_b])

def SingleHSV2RGB(hsv_color):
    hsv_color = np.reshape(hsv_color, (1, 1, 3))
    hsv_color = cv2.normalize(src=hsv_color, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    return (int(rgb_color[0,0,0]), int(rgb_color[0,0,1]), int(rgb_color[0,0,2]))


def FindPartition(img, start_x, end_x, start_y, end_y, inc):
    h, w = img.shape[:2]
    max_diff = 0
    best_partition = 1
    for partition in range(start_y+1, end_y - 1, inc):
        A = img[:partition, start_x:end_x]
        B = img[partition:, start_x:end_x]
        avg_rgb_a = AvgRGB(A)
        avg_rgb_b = AvgRGB(B)
        diff_of_avg_rgbs = np.sum(np.abs(avg_rgb_a - avg_rgb_b))
        if diff_of_avg_rgbs > max_diff:
            max_diff = diff_of_avg_rgbs
            best_partition = partition
    return best_partition




if __name__ == '__main__':
    cap = cv2.VideoCapture('vids/test2.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            h, w = frame.shape[:2]
            start_x = 0
            end_x = w
            start_y = 0
            end_y = h
            inc = 60
            partition = FindPartition(frame, start_x, end_x, start_y, end_y, inc)
            A = frame[:partition, :]
            B = frame[partition:, :]
            avg_rgb_A = AvgRGB(A)
            avg_rgb_B = AvgRGB(B)
            diff_of_avg_rgbs = np.sum(np.abs(avg_rgb_A - avg_rgb_B))
            ptsA = np.array([[w-100, 100], [w, 100], [w, 0], [w-100, 0]])
            ptsB = np.array([[w - 100, h - 100], [w, h - 100], [w, h], [w - 100, h]])
            cv2.drawContours(frame, [ptsA], 0, avg_rgb_A, -1)
            cv2.drawContours(frame, [ptsB], 0, avg_rgb_B, -1)
            cv2.line(frame, (0, partition), (w, partition), (100, 100, 255), 1) # partition
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 200, 0), 2) # bounds
            cv2.putText(frame, str(partition), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()

    cv2.destroyAllWindows()

