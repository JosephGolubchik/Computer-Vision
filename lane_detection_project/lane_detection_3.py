import cv2
import numpy as np
import math


def compute_hsv_white_yellow_binary(img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # Compute a binary thresholded image where yellow is isolated from HLS components
    img_hls_yellow_bin = np.zeros_like(hsv_img[:, :, 0])
    img_hls_yellow_bin[((hsv_img[:, :, 0] >= 40 / 360 * 255) & (hsv_img[:, :, 0] <= 50 / 360 * 255))
                       & ((hsv_img[:, :, 1] >= 80 / 100 * 255) & (hsv_img[:, :, 1] <= 255))
                       & ((hsv_img[:, :, 2] >= 90 / 100 * 255) & (hsv_img[:, :, 2] <= 255))
                       ] = 1

    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hsv_img[:, :, 0])
    img_hls_white_bin[((hsv_img[:, :, 0] >= 0) & (hsv_img[:, :, 0] <= 255))
                      & ((hsv_img[:, :, 1] >= 0) & (hsv_img[:, :, 1] <= 25 / 100 * 255))
                      & ((hsv_img[:, :, 2] >= 90 / 100 * 255) & (hsv_img[:, :, 2] <= 255))
                      ] = 1

    # Now combine both
    img_hls_white_yellow_bin = np.zeros_like(hsv_img[:, :, 0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1
    img_hls_white_yellow_bin = cv2.normalize(src=img_hls_white_yellow_bin, dst=None, alpha=0, beta=255,
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_hls_white_yellow_bin


def compute_hsv_white_binary(img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hsv_img[:, :, 0])
    img_hls_white_bin[((hsv_img[:, :, 0] >= 0) & (hsv_img[:, :, 0] <= 255))
                      & ((hsv_img[:, :, 1] >= 0) & (hsv_img[:, :, 1] <= 10 / 100 * 255))
                      & ((hsv_img[:, :, 2] >= 50 / 100 * 255) & (hsv_img[:, :, 2] <= 255))
                      ] = 1

    img_hls_white_bin = cv2.normalize(src=img_hls_white_bin, dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_hls_white_bin

def apply_mask(img, partition):
    h, w = img.shape[:2]
    mask = np.zeros(img.shape)
    mask[int(h * partition):, :] = 1
    masked = mask * img / 255
    masked = cv2.normalize(src=masked, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return masked

def line_groups(lines):
    lines2 = []
    for line in lines:
        for rho, theta in line:
            lines2.append(tuple((rho, theta)))
    lines2.sort()

    sorted_lines = [[]]
    group = 0
    hor_deg_thresh = 20
    ver_deg_thresh = 20
    group_thresh = 100

    can_continue = False
    lines = lines2
    i = 0
    while not can_continue:
        if (np.deg2rad(90 + hor_deg_thresh) < lines[i][1] < np.deg2rad(180 + ver_deg_thresh)) or (np.deg2rad(ver_deg_thresh) < lines[i][1] < np.deg2rad(90 - hor_deg_thresh)):
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
    avg_lines = [[]]
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
            avg_lines[0].append([rho_avg, theta_median])

    return avg_lines, best_two_avg_lines(avg_lines)



def draw_lines(img, lines, color=(0,0,255)):
    img_lines = img.copy()
    if lines != [[]]:
        points = []
        for line in lines:
            a = np.cos(line[1])
            b = np.sin(line[1])
            x0 = a * line[0]
            y0 = b * line[0]
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

def LaneDetection(img, prev_best_two=None):
    # img = cv2.GaussianBlur(img, (5,5), -1)
    hsv = compute_hsv_white_binary(img)
    edges = cv2.Canny(hsv, 50, 200, apertureSize=3)
    mask = cv2.imread('img/mask2.png',0)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    edges = (mask / 255) * edges
    edges = cv2.normalize(src=edges, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)
    if lines is None:
        return img, None
    avg_lines, best_two = avg_of_line_groups(lines)
    prob = 0.88
    if prev_best_two is not None:
        diff_best_two = 0
        for i in range(2):
            diff_best_two += np.abs(prev_best_two[i][0] - best_two[i][0])
        if diff_best_two > 50:
            best_two = prev_best_two
        else:
            for i in range(2):
                best_two[i] = [best_two[i][0] * (1-prob) + prev_best_two[i][0] * prob, best_two[i][1] * (1-prob) + prev_best_two[i][1] * prob]
    img_lines = draw_lines(img, best_two)
    img_rect = draw_lane_rect(img, best_two)
    return img_rect, best_two

# def create_mask(img):
#

if __name__ == '__main__':
    cap = cv2.VideoCapture('vids/test1.mp4')
    # out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP42'), 15, (1280, 720))

    best_two = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            if best_two is None:
                frame, best_two = LaneDetection(frame)
            else:
                frame, best_two = LaneDetection(frame, best_two)

            # mask = cv2.imread('img/mask2.png')
            # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            # frame = (mask / 255) * frame
            # frame = cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            cv2.imshow('frame', frame)
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
            # out.release()

    cv2.destroyAllWindows()
