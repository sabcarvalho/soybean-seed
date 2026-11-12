"""Image preprocessing helpers."""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def prepare_img(img, roi=(30, 600, 200, 650)):
    """Zero out pixels outside the region of interest (y0, y1, x0, x1)."""
    y0, y1, x0, x1 = roi
    mask = np.ones_like(img, dtype=bool)
    mask[y0:y1, x0:x1] = False
    copy = img.copy()
    copy[mask] = 0
    return copy


def plot_homography(img_rgb, homographed_rx, roi=(30, 600, 200, 650)):
    crop = prepare_img(img_rgb, roi)
    blended = cv2.addWeighted(crop, 0.5, homographed_rx, 0.5, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Sobreposição após Homografia")
    plt.axis("off")
    plt.show()