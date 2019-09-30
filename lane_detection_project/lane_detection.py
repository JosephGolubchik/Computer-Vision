import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import tracemalloc


def func(img):
    mask = cv2.imread('img/mask.jpeg', 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), -1)
    # mask = np.zeros(gray.shape)
    # mask[int(gray.shape[0] * 0.5):, :] = 1
    mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))
    edges = cv2.Canny(gray, 100, 300)
    # edges = (mask / 255) * edges
    edges = cv2.normalize(src=edges, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    # print(lines.shape)
    # for line in lines:
    #     for rho,theta in line:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 4000*(-b))
    #         y2 = int(y0 - 4000*(a))
    #
    #         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    if lines is None:
        return img
    lines2 = []
    for line in lines:
        for rho, theta in line:
            lines2.append(tuple((rho, theta)))
    lines2.sort()
    # for line in lines:
    #     for rho,theta in line:
    #         if theta > np.deg2rad(90 + 25) or theta < np.deg2rad(90 - 25):
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 4000*(-b))
    #             y2 = int(y0 - 4000*(a))
    #
    #             cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    sorted_lines = [[]]
    group = 0
    thresh = 15
    group_thresh = 150
    for i in range(len(lines2) - 2):
        if lines2[i][1] > np.deg2rad(90 + thresh) or lines2[i][1] < np.deg2rad(90 - thresh):
            if np.abs(lines2[i][0] - lines2[i + 1][0]) < group_thresh:
                sorted_lines[group].append(lines2[i])
            else:
                group += 1
                sorted_lines.append([])
                sorted_lines[group].append(lines2[i])

    if lines2[len(lines2) - 1][1] > np.deg2rad(90 + thresh) or lines2[len(lines2) - 1][1] < np.deg2rad(90 - thresh):
        if np.abs(lines2[len(lines2) - 1][0] - lines2[len(lines2) - 2][0]) < group_thresh:
            sorted_lines[group].append(lines2[len(lines2) - 1])
        else:
            group += 1
            sorted_lines.append([])
            sorted_lines[group].append(lines2[len(lines2) - 1])

    # sorted_lines[0].append(lines2[0])
    # prev_rho = lines2[0][0]
    # group = 0
    # thresh = 25
    # for line in lines2[1:]:
    #     if line[1] > np.deg2rad(90 + thresh) or line[1] < np.deg2rad(90 - thresh):
    #         if np.abs(line[0] - prev_rho) < 40:
    #             sorted_lines[group].append(line)
    #         else:
    #             group += 1
    #             sorted_lines.append([])
    #             sorted_lines[group].append(line)
    #         prev_rho = line[0]

    avg_lines = []
    for line_group in sorted_lines:
        if len(line_group) != 0:
            # print(line_group)
            # rho_sum = 0
            # theta_sin_sum = 0
            # theta_cos_sum = 0
            # for line in line_group:
            # rho_sum += line[0]
            # theta_sin_sum += np.sin(line[1])
            # theta_cos_sum += np.cos(line[1])
            # rho_avg = rho_sum / len(line_group)
            n = len(line_group)
            rho_avg = line_group[n // 2][0]
            line_group.sort(key=lambda a: a[1])
            theta_avg = line_group[n // 2][1]
            # theta_avg = math.atan2((1/n) * theta_sin_sum, (1/n) * theta_cos_sum)
            # print(np.rad2deg(theta_avg))
            avg_lines.append((rho_avg, theta_avg))
    img_lines = img.copy()
    # print(len(avg_lines))
    points = []

    if len(avg_lines) >= 2:
        best_two = [avg_lines[0], avg_lines[len(avg_lines)-1]]

        for line in best_two:
            a = np.cos(line[1])
            b = np.sin(line[1])
            x0 = a * line[0]
            y0 = b * line[0]
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 4000 * (-b))
            y2 = int(y0 - 4000 * (a))
            m = (y2 - y1) / (x2 - x1)
            b = y1 - (m * x1)
            points.append((m, b))
            # cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # print("xy = "+str(str(x0)+" "+str(-1*y0)))
            # print("ab = "+str(str(a)+" "+str(b)))
            # print("line = "+str(str(line[0])))
            # print("p0 = "+str(str(x0)+" "+str(y0)))
            # print("p1 = "+str(str(x1)+" "+str(y1))+", p2 = "+str(str(x2)+" "+str(y2)))
        if len(points) >= 2:
            p1 = [int((img.shape[0] - points[0][1]) // points[0][0]), int(img.shape[0])]
            p2 = [int((img.shape[0] - points[1][1]) // points[1][0]), int(img.shape[0])]
            xc = int((points[1][1] - points[0][1]) // (points[0][0] - points[1][0] +0.000001))
            yc = int(points[0][0] * xc + points[0][1])
            p3 = [xc, yc]
            pts = np.array([p1, p2, p3])
            road_mask = img_lines.copy()
            cv2.drawContours(road_mask, [pts], 0, (0, 255, 0, 100), -1)
            road_mask[0:yc+int(img.shape[0]*0.1), :] = img[0:yc+int(img.shape[0]*0.1), :]
            img_lines = img_lines*0.7 + road_mask*0.3
            img_lines = cv2.normalize(src=img_lines, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_lines


if __name__ == '__main__':
    cap = cv2.VideoCapture('lane9.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            res = func(frame)

            cv2.imshow('frame', res)
            time.sleep(0.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()

    cv2.destroyAllWindows()
