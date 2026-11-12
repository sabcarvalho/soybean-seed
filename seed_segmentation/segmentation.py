"""Segmentation of RGB and RX images."""
from math import pi
import cv2
import numpy as np
from skimage import morphology

from .preprocessing import prepare_img


def calculate_morphology(erosion_radius, dilation_radius, img):
    """Otsu threshold → erosion → dilation → uint8 {0, 255}."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    disk = morphology.disk(erosion_radius)
    eroded = morphology.erosion(thresh, disk)

    disk = morphology.disk(dilation_radius)
    eroded_dilated = morphology.dilation(eroded, disk)

    converted = np.array(eroded_dilated, dtype=int)
    converted = cv2.normalize(converted, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return converted


def segmentation_rgb(img, cfg, roi=(30, 600, 200, 650)):
    crop = prepare_img(img, roi)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    result = calculate_morphology(cfg.rgb_erosion, cfg.rgb_dilation, saturation)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = (4 * pi * area) / (perim ** 2)
        if circ > cfg.min_circularity:
            filtered.append(cnt)
    return tuple(filtered)


def segmentation_rx(img, cfg, rotate=True):
    copy = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if rotate else img.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    result = calculate_morphology(cfg.rx_erosion, cfg.rx_dilation, gray)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [c for c in contours if cv2.contourArea(c) > cfg.min_area]
    return tuple(filtered)