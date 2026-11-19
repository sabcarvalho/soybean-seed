"""Centroids, point ordering and homography."""
import cv2
import numpy as np


def calculate_centroids(contours):
    centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append([cX, cY])
    return centroids


def ordenate_points(points, contours=None, columns_per_row=5):
    """Order points row-by-row (top to bottom, left to right)."""
    if contours is None:
        y_ordered = sorted(points, key=lambda p: p[0][1])
    else:
        y_ordered = sorted(zip(points, contours), key=lambda p: p[0][1])

    ordered = []
    rows = len(y_ordered) // columns_per_row
    i = 0
    for _ in range(rows):
        row = y_ordered[i:i + columns_per_row]
        row = sorted(row, key=lambda p: p[0][0])
        ordered.extend(row)
        i += columns_per_row

    if contours is None:
        return ordered
    cents, conts = zip(*ordered)
    return cents, conts


def homography(img_rx, centroids_rx, img_rgb, centroids_rgb,
               ransac_threshold=1.0):
    c_rgb = np.array(centroids_rgb)
    c_rx = np.array(centroids_rx)
    M, _ = cv2.findHomography(c_rx, c_rgb, cv2.RANSAC, ransac_threshold)

    copy_rx = cv2.rotate(img_rx, cv2.ROTATE_90_CLOCKWISE)
    h, w = img_rgb.shape[:2]
    return cv2.warpPerspective(copy_rx, M, (w, h))