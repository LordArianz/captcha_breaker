from cv2 import cv2
import numpy as np
from config import CONFIGURATION


def mask(img, low, high, new_color=(0, 0, 0)):
    low = np.array(low)
    high = np.array(high)

    mask = cv2.inRange(img, low, high)

    img[mask > 0] = new_color
    return img


def split_letters(fname, num_letters=6, debug=False):
    raw_image = cv2.imread(fname)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    mask(raw_image, [200, 150, 0], [255, 190, 50], (255, 255, 255))
    mask(raw_image, [200, 0, 0], [255, 80, 80])
    mask(raw_image, [0, 100, 0], [100, 200, 100])

    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)

    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    weighted_image = cv2.addWeighted(gray_image, 1.80, blur_image, -0.55, 0)

    not_image = cv2.bitwise_not(weighted_image)

    _, thresh_image = cv2.threshold(not_image, 150, 255, cv2.THRESH_BINARY)

    height, width = np.shape(thresh_image)
    points = np.zeros((int(cv2.sumElems(thresh_image)[0]), 2))

    z = 0
    for x in range(width):
        for y in range(height):
            if thresh_image[y][x] == 255:
                points[z][0] = y
                points[z][1] = x
                z += 1

    points = np.float32(points)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(1e8), 1e-4)
    attempts = 10
    _, _, centers = cv2.kmeans(points, num_letters + 1, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # seg = thresh_image.copy()
    centers = sorted(centers[1:], key=lambda x: x[1])
    letters = []
    for center in centers:
        x = int(center[1])
        y = int(center[0])
        hl_x = CONFIGURATION['data']['width'] // 2
        hl_y = CONFIGURATION['data']['height'] // 2
        
        letter = thresh_image[y - hl_y: y + hl_y, x - hl_x: x + hl_x].copy()
        letters.append(letter)
        # seg = cv2.circle(seg, (x, y), 2, (255, 0, 0))
        # seg = cv2.rectangle(seg, (x - hl_x, y - hl_y), (x + hl_x, y + hl_y), (255, 0, 0))

    return letters
