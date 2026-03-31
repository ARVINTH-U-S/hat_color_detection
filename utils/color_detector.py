import cv2
import numpy as np

color_ranges = {
    "Grey": [([0, 0, 50], [180, 50, 125])],
    "White": [([0, 0, 200], [180, 40, 255])],
    "Green": [([35, 50, 50], [85, 255, 255])],
    "Yellow": [([20, 100, 100], [35, 255, 255])],
    "Blue": [([90, 80, 50], [130, 255, 255])],
    "Red": [
        ([0, 100, 100], [10, 255, 255]),
        ([160, 100, 100], [180, 255, 255])
    ],
    "Brown": [([10, 100, 20], [20, 255, 200])]
}

def threshold_color_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_pixel_counts = {}
    for color_name, ranges in color_ranges.items():
        mask = None
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            part_mask = cv2.inRange(hsv_image, lower, upper)
            mask = part_mask if mask is None else cv2.bitwise_or(mask, part_mask)
        count = cv2.countNonZero(mask)
        if count > 0:
            color_pixel_counts[color_name] = count
    if color_pixel_counts:
        max_color = max(color_pixel_counts, key=color_pixel_counts.get)
        return max_color, color_pixel_counts[max_color]
    else:
        return None, 0