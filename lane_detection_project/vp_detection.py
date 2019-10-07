import numpy as np
import cv2
import itertools
import random
from itertools import starmap

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

# Perform edge detection
def hough_transform(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    # kernel = np.ones((9, 9), np.uint8)

    # opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    # edges = cv2.Canny(opening, 50, 150, apertureSize=3)  # Canny edge detection

    hsv = compute_hsv_white_binary(img)
    edges = cv2.Canny(hsv, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)  # Hough line detection

    hough_lines = []
    hor_deg_thresh = 20
    ver_deg_thresh = 15
    # Lines are represented by rho, theta; converted to endpoint notation
    if lines is not None:
        for line in lines:
            if (np.deg2rad(90 + hor_deg_thresh) < line[0][1] < np.deg2rad(180 + ver_deg_thresh)) or (
                    np.deg2rad(ver_deg_thresh) < line[0][1] < np.deg2rad(90 - hor_deg_thresh)):
                hough_lines.extend(list(starmap(endpoints, line)))

    return hough_lines


def endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x_0 = a * rho
    y_0 = b * rho
    x_1 = int(x_0 + 1000 * (-b))
    y_1 = int(y_0 + 1000 * (a))
    x_2 = int(x_0 - 1000 * (-b))
    y_2 = int(y_0 - 1000 * (a))

    return ((x_1, y_1), (x_2, y_2))


# Random sampling of lines
def sample_lines(lines, size):
    if size > len(lines):
        size = len(lines)
    return random.sample(lines, size)


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Lines don't cross

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y


# Find intersections between multiple lines (not line segments!)
def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    return intersections


# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img, grid_size, intersections):
    # Image dimensions
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Grid dimensions
    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = (0.0, 0.0)
    best_sum = (0, 0)
    avg_vp = (0,0)

    # for intersection in intersections:
        # avg_vp = (int((avg_vp[0] + intersection[0])//2), int((avg_vp[1] + intersection[1])//2))

    for i, j in itertools.product(range(grid_columns),range(grid_rows)):
        cell_left = i * grid_size
        cell_right = (i + 1) * grid_size
        cell_bottom = j * grid_size
        cell_top = (j + 1) * grid_size
        # cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 1)

        current_intersections = 0  # Number of intersections in the current cell
        current_sum = (0, 0)
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1
                current_sum = (current_sum[0] + x, current_sum[1] + y)

        # Current cell has more intersections that previous cell (better)
        if current_intersections > max_intersections:
            max_intersections = current_intersections
            best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
            best_sum = current_sum

            # print("Best Cell:", best_cell)

    avg_vp = (int(best_sum[0]//(max_intersections+0.001)), int(best_sum[1]//(max_intersections+0.001)))

    # if best_cell[0] != None and best_cell[1] != None:
    #     rx1 = int(best_cell[0] - grid_size / 2)
    #     ry1 = int(best_cell[1] - grid_size / 2)
    #     rx2 = int(best_cell[0] + grid_size / 2)
    #     ry2 = int(best_cell[1] + grid_size / 2)
    #     cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)

    return best_cell, avg_vp

if __name__ == '__main__':
    cap = cv2.VideoCapture('vids/test1.mp4')
    prev_avg_vp = None
    avg_vp = (0,0)
    prob = 0.99
    while (cap.isOpened()):
        ret, frame = cap.read()

        x_bound = 0
        y_start = frame.shape[0] // 2

        if ret is True:

            hough_lines = hough_transform(frame[y_start:, x_bound:frame.shape[1]-x_bound, :])
            if hough_lines:
                print(len(hough_lines))
                random_sample = sample_lines(hough_lines, 300)
                intersections = find_intersections(random_sample)
                if intersections:
                    grid_size = min(frame.shape[0], frame.shape[1]) // 5
                    vanishing_point, avg_vp = find_vanishing_point(frame, grid_size, intersections)
                    avg_vp = (avg_vp[0] + x_bound, avg_vp[1] + y_start)
                    if prev_avg_vp is not None:
                        avg_vp = (int(prev_avg_vp[0]*prob + avg_vp[0]*(1-prob)), int(prev_avg_vp[1]*prob + avg_vp[1]*(1-prob)))
                    prev_avg_vp = avg_vp
                elif prev_avg_vp is not None:
                    avg_vp = prev_avg_vp
            elif prev_avg_vp is not None:
                avg_vp = prev_avg_vp
            cv2.circle(frame, avg_vp, 0, (0, 255, 0), 10)
            # cv2.line(frame, (avg_vp[0]-avg_vp[1]//15, avg_vp[1]+30), (avg_vp[0]+avg_vp[1]//15, avg_vp[1]+30), (255,255,255), 2)
            road_mask = np.zeros(frame.shape)
            pts = np.array([(frame.shape[1]//5, frame.shape[0]), (avg_vp[0]-avg_vp[1]//15, avg_vp[1]+30), (avg_vp[0]+avg_vp[1]//15, avg_vp[1]+30), (frame.shape[1] - frame.shape[1]//5, frame.shape[0])])
            cv2.drawContours(road_mask, [pts], 0, (255, 255, 255), -1)
            frame = (road_mask / 255) * frame
            frame = cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
    cv2.destroyAllWindows()
