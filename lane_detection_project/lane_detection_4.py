import cv2
import numpy as np
import matplotlib.pyplot as plt


def line_groups(lines):
    lines2 = []
    for line in lines:
        for rho, theta in line:
            lines2.append(tuple((rho, theta)))
    lines2.sort()

    sorted_lines = [[]]
    group = 0
    group_thresh = 100

    can_continue = False
    lines = lines2
    i = 0
    while not can_continue:
        sorted_lines[group].append([lines[i]])
        can_continue = True
        prev_line = lines[i]
        i += 1
        if i == len(lines):
            return sorted_lines

    for line in lines[i+1:]:
        if np.abs(line[0] - prev_line[0]) < group_thresh:
            sorted_lines[group].append([line])
        else:
            group += 1
            sorted_lines.append([])
            sorted_lines[group].append([line])
        prev_line = line

    return sorted_lines


def best_two_avg_lines(avg_lines):
    if len(avg_lines[0][0]) < 2:
        best_two = [[[300,0]], [[300,0]]]
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
        best_two = [[best_two[0]], [best_two[1]]]
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
    if len(img.shape) < 3:
        img_lines = np.zeros((img.shape[0], img.shape[1], 3))
        img_lines[:, :, 0] = img
        img_lines[:, :, 1] = img
        img_lines[:, :, 2] = img
    else:
        img_lines = img.copy()
    if lines != [[]]:
        points = []
        for line in lines:
            print(np.array(line))
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


def hough_lines(img, hor_threshold, ver_threshold):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 70)
    sorted_lines = []
    if lines is not None:
        for line in lines:
            if (np.deg2rad(90 + hor_threshold) < line[0][1] < np.deg2rad(180 - ver_threshold)) or (
                    np.deg2rad(ver_threshold) < line[0][1] < np.deg2rad(90 - hor_threshold)):
                sorted_lines.append(line)
    return sorted_lines

def hough_linesZ(img, hor_threshold, ver_threshold):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 70)
    sorted_lines = []
    if lines is not None:
        for line in lines:
            if (np.deg2rad(90 + hor_threshold) < line[0][1] < np.deg2rad(180 - ver_threshold)) or (
                    np.deg2rad(ver_threshold) < line[0][1] < np.deg2rad(90 - hor_threshold)):
                sorted_lines.append(line)
    return sorted_lines


def draw_hough_lines(img, hor_threshold, ver_threshold, color=None):
    lines = hough_lines(img, hor_threshold, ver_threshold)
    avg_lines = avg_of_line_groups(lines)
    # best_two = best_two_avg_lines(avg_lines)
    # print(len(best_two))
    # img_lines = draw_lines(img, lines, (0, 255, 0))
    img_lines = draw_lines(img, avg_lines, (255, 0, 0))
    # img_lines = draw_lines(img_lines, best_two)
    return img_lines


def only_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hls_white_bin = np.zeros_like(hsv[:, :, 0])
    img_hls_white_bin[((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 255))
                      & ((hsv[:, :, 1] >= 0) & (hsv[:, :, 1] <= 10/100*255))
                      & ((hsv[:, :, 2] >= 80 / 100 * 255) & (hsv[:, :, 2] <= 255))
                      ] = 1

    img_hls_white_bin = cv2.normalize(src=img_hls_white_bin, dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_hls_white_bin


def morph(img, flag, shape):
    return cv2.morphologyEx(img, flag, np.ones(shape))


def otsu(img):
    res = cv2.cvtColor(img, cv2. COLOR_RGB2GRAY)
    ret, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res


def canny(img, low, high):
    return cv2.Canny(img, low, high)


def perspective_transform(img):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[420, 720],
        [780, 720],
        [540, 560],
        [720, 560]])
    dst = np.float32(
        [[300, 720],
        [980, 720],
        [300, 0],
        [980, 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    return warped, unwarped, m, m_inv

def row_hist(img):
    hist = np.zeros(img.shape[1])
    for row in img:
        hist += row
    max = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
    max1 = 0
    max2 = 0
    maxi1 = 0
    maxi2 = 0
    for i in np.arange(1, hist.size - 1)[max]:
        if hist[i] > max1:
            max1 = hist[i]
            maxi1 = i
        elif hist[i] > max2:
            max2 = hist[i]
            maxi2 = i
    return (maxi1, maxi2)

def sliding_window(img):
    hist = row_hist(img)


    h, w = img.shape[:2]
    # for i in range(h):
    #     for j in range(w):



def process_frame(frame, func, args=None):
    if args is not None:
        return func(frame, *args)
    else:
        return func(frame)


def main():
    cap = cv2.VideoCapture('vids/test2.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        img = frame
        if ret:
            # frame = process_frame(frame, otsu)
            # frame = process_frame(frame, only_white)
            # frame = process_frame(frame, morph, (cv2.MORPH_OPEN, (2,2)))
            frame = perspective_transform(frame)[0]
            # frame = process_frame(frame, canny, (100, 150))
            # frame = process_frame(frame, draw_hough_lines, (0, 0))
            # frame[frame.shape[0]-20:,:] = 0
            # max1, max2 = row_hist(frame)
            # cv2.line(frame, (max1, frame.shape[0]), (max1, 0), (255,0,0), 2)
            # cv2.line(frame, (max2, frame.shape[0]), (max2, 0), (255, 0, 0), 2)


            cv2.imshow('Lanes', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
