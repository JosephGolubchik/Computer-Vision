import numpy as np
import cv2


def line_groups(lines):
    lines2 = []
    for line in lines:
        for rho, theta in line:
            lines2.append(tuple((rho, theta)))
    lines2.sort()

    sorted_lines = [[]]
    group = 0
    hor_deg_thresh = 15
    ver_deg_thresh = 20
    group_thresh = 100

    can_continue = False
    lines = lines2
    i = 0
    while not can_continue:
        if (np.deg2rad(90 + hor_deg_thresh) < lines[i][1] < np.deg2rad(180 - ver_deg_thresh)) or (np.deg2rad(ver_deg_thresh) < lines[i][1] < np.deg2rad(90 - hor_deg_thresh)):
            sorted_lines[group].append([lines[i]])
            can_continue = True
            prev_line = lines[i]
        i += 1
        if i == len(lines):
            return sorted_lines

    for line in lines[i+1:]:
        if (np.deg2rad(90 + hor_deg_thresh) < line[1] < np.deg2rad(180 + ver_deg_thresh)) or (
                np.deg2rad(ver_deg_thresh) < line[1] < np.deg2rad(90 - hor_deg_thresh)):
            if np.abs(line[0] - prev_line[0]) < group_thresh:
                sorted_lines[group].append([line])
            else:
                group += 1
                sorted_lines.append([])
                sorted_lines[group].append([line])
            prev_line = line

    # if lines[0, 0, 1] > np.deg2rad(90 + deg_thresh) or lines[0, 0, 1] < np.deg2rad(90 - deg_thresh):
    #     sorted_lines[group].append([lines[0, 0]])
    #
    # if lines.shape[0] > 1:
    #     for i in range(1, lines.shape[0]):
    #         if lines[i, 0, 1] > np.deg2rad(90 + deg_thresh) or lines[i, 0, 1] < np.deg2rad(90 - deg_thresh):
    #             if np.abs(lines[i, 0, 0] - lines[i - 1, 0, 0]) < group_thresh:
    #                 sorted_lines[group].append([lines[i, 0]])
    #             else:
    #                 group += 1
    #                 sorted_lines.append([])
    #                 sorted_lines[group].append([lines[i, 0]])
    return sorted_lines

def best_two_avg_lines(avg_lines):
    if len(avg_lines[0]) < 2:
        best_two = [[300,0], [300,0]]
    else:
        min_diff1 = 100
        min_diff2 = 200
        best_line1 = avg_lines[0][0]
        best_line2 = avg_lines[0][0]
        for line in avg_lines[0]:
            theta_diff_from_vertical = min(line[1], np.pi - line[1])
            if theta_diff_from_vertical < min_diff1:
                min_diff1 = theta_diff_from_vertical
                best_line1 = line
            elif theta_diff_from_vertical < min_diff2:
                min_diff2 = theta_diff_from_vertical
                best_line2 = line
        best_two = [best_line1, best_line2]
        best_two.sort(key=lambda x:x[0])
    return best_two

def avg_of_line_groups(lines):
    sorted_lines = line_groups(lines)
    avg_lines = []
    for line_group in sorted_lines:
        if len(line_group) != 0:
            rho_sum = 0
            theta_sin_sum = 0
            theta_cos_sum = 0
            theta_sum = 0
            for line in line_group:
                rho_sum += line[0][0]
                theta_sin_sum += np.sin(line[0][1])
                theta_cos_sum += np.cos(line[0][1])
                theta_sum += line[0][1]
            rho_avg = rho_sum / len(line_group)
            n = len(line_group)
            # rho_avg = line_group[n // 2][0]
            line_group.sort(key=lambda x:x[0][1])
            theta_median = line_group[n // 2][0][1]
            # theta_avg = math.atan2((1 / n) * theta_sin_sum, (1 / n) * theta_cos_sum)
            # theta_avg = theta_sum / len(line_group)
            avg_lines.append([[rho_avg, theta_median]])
    return avg_lines

def draw_lines(img, lines, color=(0,0,255)):
    img_lines = img.copy()
    if lines != [[]]:
        points = []
        for line in lines:
            a = np.cos(line[0][1])
            b = np.sin(line[0][1])
            x0 = a * line[0][0]
            y0 = b * line[0][0]
            x1 = int(x0 + 4000 * (-b))
            y1 = int(y0 + 4000 * (a))
            x2 = int(x0 - 4000 * (-b))
            y2 = int(y0 - 4000 * (a))
            m = (y2 - y1) / (x2 - x1 + 0.001)
            b = y1 - (m * x1)
            points.append((m, b))
            cv2.line(img_lines, (x1, y1), (x2, y2), color, 2)
    return img_lines

def draw_lane_rect(img, best_two):
    img_rect = img.copy()
    points = []
    for line in best_two:
        a = np.cos(line[1])
        b = np.sin(line[1])
        x0 = a * line[0]
        y0 = b * line[0]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 4000 * (-b))
        y2 = int(y0 - 4000 * (a))
        m = (y2 - y1) / (x2 - x1 + 0.000000001)
        b = y1 - (m * x1)
        points.append((m, b))
        # cv2.line(img_rect, (x1, y1), (x2, y2), (0, 0, 255), 2)

    p1 = [int((img.shape[0] - points[0][1]) // points[0][0]), int(img.shape[0])]
    p2 = [int((img.shape[0] - points[1][1]) // points[1][0]), int(img.shape[0])]
    xc = int((points[1][1] - points[0][1]) // (points[0][0] - points[1][0] +0.000001))
    yc = int(points[0][0] * xc + points[0][1])
    p3 = [xc, yc]
    pts = np.array([p1, p2, p3])
    road_mask = img_rect.copy()
    cv2.drawContours(road_mask, [pts], 0, (0, 255, 0, 100), -1)
    road_mask[0:yc+int(img.shape[0]*0.05), :] = img[0:yc+int(img.shape[0]*0.05), :]
    img_rect = img_rect*0.8 + road_mask*0.2
    img_rect = cv2.normalize(src=img_rect, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_rect

def hough_lines(edges, hor_threshold, ver_threshold):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 70)
    sorted_lines = []
    if lines is not None:
        for line in lines:
            if (np.deg2rad(90 + hor_threshold) < line[0][1] < np.deg2rad(180 - ver_threshold)) or (
                    np.deg2rad(ver_threshold) < line[0][1] < np.deg2rad(90 - hor_threshold)):
                sorted_lines.append(line)
    print(np.array(lines).shape, np.array(sorted_lines).shape)
    return sorted_lines

def LaneDetection(curr, prvs):
    # prvs[:prvs.shape[0]//3,:] = 0
    # curr[:prvs.shape[0]//3,:] = 0
    hsv = np.zeros_like(prvs)
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
    hsv[..., 1] = 255
    # flow = cv2.calcOpticalFlowFarneback(prvs, curr, None, 0.5, 3, 20, 3, 1, 1, 0)
    #
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # mask = np.zeros(rgb.shape)
    # print(np.argwhere(rgb == 0))
    # mask[np.argwhere(rgb == 0)] = 1
    # gray[gray == 0] = 255
    # gray[gray != 255] = 0
    # gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, np.ones((21,21)))

    blur = cv2.GaussianBlur(curr, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.normalize(src=th, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    th[:400,:] = 0
    edges = cv2.Canny(th, 100, 150, apertureSize=3)
    lines = hough_lines(edges, 30, 30)
    if lines is None:
        return th
    # avg_lines = avg_of_line_groups(lines)

    img_lines = draw_lines(curr, lines)
    # img_rect = draw_lane_rect(img, avg_lines)
    # mask = cv2.imread('img/mask4.png')
    # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    # img_rect = img_rect * mask
    return img_lines


if __name__ == '__main__':

    cap = cv2.VideoCapture("vids/test2.mp4")
    ret, frame1 = cap.read()
    prvs = frame1

    while (cap.isOpened()):
        ret, frame2 = cap.read()
        if ret is True:
            curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            curr = LaneDetection(curr, prvs)
            cv2.imshow('curr', curr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()

    cv2.destroyAllWindows()

    # while (1):
    #     ret, frame2 = cap.read()
    #
    #
    #     edges = cv2.Canny(rgb, 50, 150, apertureSize=3)
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # Hough line detection
    #     avg_lines, best_two = avg_of_line_groups(lines)
    #     prob = 0.5
    #     if prev_best_two is not None:
    #         diff_best_two = 0
    #         for i in range(2):
    #             diff_best_two += np.abs(prev_best_two[i][0] - best_two[i][0])
    #         if diff_best_two > 100:
    #             best_two = prev_best_two
    #         else:
    #             for i in range(2):
    #                 best_two[i] = [best_two[i][0] * (1 - prob) + prev_best_two[i][0] * prob,
    #                                best_two[i][1] * (1 - prob) + prev_best_two[i][1] * prob]
    #
    #     cv2.imshow('frame2', rgb)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #     elif k == ord('s'):
    #         cv2.imwrite('opticalfb.png', frame2)
    #         cv2.imwrite('opticalhsv.png', rgb)
    #     prvs = next
    #
    # cap.release()
    # cv2.destroyAllWindows()