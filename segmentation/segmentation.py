"""Segmentation of RGB and RX images."""
from math import pi
import cv2
import numpy as np
from skimage import morphology

from .preprocessing import prepare_img


def retry_segment_slot(img, cx, cy, mode="rgb",
                        roi_size=80,
                        threshold_factor=0.6,
                        erosion=1,
                        min_area=None,
                        max_offset=None,
                        verbose=False):
    H, W = img.shape[:2]
    half = roi_size // 2

    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return None

    roi = img[y0:y1, x0:x1]

    # ---- feature ----
    if mode == "rgb":
        feature = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    else:  # rx / grayscale
        feature = (roi if roi.ndim == 2
                   else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    # ---- threshold relaxado ----
    otsu_val, _ = cv2.threshold(feature, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    relaxed = max(1, int(otsu_val * threshold_factor))
    _, binary = cv2.threshold(feature, relaxed, 255, cv2.THRESH_BINARY)

    # ---- erosão mínima ----
    if erosion > 0:
        k = 2 * erosion + 1
        binary = cv2.erode(binary, np.ones((k, k), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # ---- filtro de área ----
    if min_area is None:
        min_area = 0.005 * (roi_size ** 2)   # ~0.5% do ROI
    candidates = [c for c in contours if cv2.contourArea(c) > min_area]
    if not candidates:
        return None

    # ---- escolha: mais próximo do centro do ROI ----
    roi_cx, roi_cy = cx - x0, cy - y0

    def _dist(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float("inf")
        return float(np.hypot(M["m10"] / M["m00"] - roi_cx,
                              M["m01"] / M["m00"] - roi_cy))

    best = min(candidates, key=_dist)

    if max_offset is None:
        max_offset = roi_size // 3
    if _dist(best) > max_offset:
        if verbose:
            print(f"  [retry] slot ({cx},{cy}) rejeitado: "
                  f"dist={_dist(best):.1f} > {max_offset}")
        return None

    # ---- converte ROI -> imagem original ----
    best = best.copy()
    best[:, :, 0] += x0
    best[:, :, 1] += y0

    if verbose:
        print(f"  [retry] slot ({cx},{cy}): OK "
              f"(área={cv2.contourArea(best):.1f}, "
              f"dist={_dist(min(candidates, key=_dist)):.1f})")
    return best

def calculate_morphology(erosion_radius, dilation_radius, img, normalization = 255):
    """Otsu threshold → erosion → dilation → uint8 {0, 255}."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    disk = morphology.disk(erosion_radius)
    eroded = morphology.erosion(thresh, disk)

    disk = morphology.disk(dilation_radius)
    eroded_dilated = morphology.dilation(eroded, disk)

    converted = np.array(eroded_dilated, dtype=int)
    converted = cv2.normalize(converted, None, 0, normalization, cv2.NORM_MINMAX, cv2.CV_8U)
    return converted


def segmentation_rgb(img, cfg, return_mask=False):
    roi = cfg.roi_rgb
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

    if return_mask:
        final_mask = np.zeros_like(result)
        cv2.drawContours(final_mask, filtered, -1, 255, thickness=-1)
        return tuple(filtered), final_mask
    return tuple(filtered)


def segmentation_rx(img, cfg, rotate=True, return_mask=False):
    roi = cfg.roi_rx
    crop = prepare_img(img, roi)
    copy = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE) if rotate else img.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    result = calculate_morphology(cfg.rx_erosion, cfg.rx_dilation, gray)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [c for c in contours if cv2.contourArea(c) > cfg.min_area]

    if return_mask:
        final_mask = np.zeros_like(result)
        cv2.drawContours(final_mask, filtered, -1, 255, thickness=-1)
        return tuple(filtered), final_mask
    return tuple(filtered)