import cv2
import numpy as np
from .geometry import ordenate_points
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from .segmentation import calculate_morphology
from skimage import morphology

def seed_mask_in_crop(crop, center_local, max_radius=None, mode="auto", median_area=None):
    if crop is None or crop.size == 0:
        return None

    # ---- auto mode ----
    if mode == "auto":
        if crop.ndim == 2:
            mode = "rx"
        else:
            sat_mean = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
            mode = "rx" if sat_mean < 5 else "rgb"

    # ---- feature ----
    if mode == "rgb":
        feature = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    else:
        feature = (crop if crop.ndim == 2
                   else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

    cx = int(np.clip(center_local[0], 0, feature.shape[1] - 1))
    cy = int(np.clip(center_local[1], 0, feature.shape[0] - 1))

    # ---- threshold + CLOSE para "colar" os pedaços da semente ----
    _, binary = cv2.threshold(feature, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mode == "rx" and binary[cy, cx] == 0:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # CLOSE com kernel proporcional ao tamanho do crop — fecha "rachaduras"
    # da saturação no meio da semente (causa B)
    kclose = 5 if min(binary.shape) >= 40 else 3
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((kclose, kclose), np.uint8))

    if binary[cy, cx] == 0:
        cv2.circle(binary, (cx, cy), 2, 255, -1)

    # ---- CC primeiro (sempre) ----
    _, labels_cc = cv2.connectedComponents(binary)
    lbl = labels_cc[cy, cx]
    if lbl == 0:
        return None
    mask_cc = ((labels_cc == lbl) * 255).astype(np.uint8)
    cc_area = int((mask_cc > 0).sum())

    # ---- watershed só quando suspeita de fusão ----
    mask = mask_cc
    if median_area is not None and cc_area > 1.6 * median_area:
        eq_diam = 2 * np.sqrt(median_area / np.pi)
        min_dist = max(8, int(0.4 * eq_diam))

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        peaks = peak_local_max(dist, min_distance=min_dist,
                               threshold_abs=3, exclude_border=False,
                               labels=binary)
        if len(peaks) >= 2:
            markers = np.zeros_like(binary, dtype=np.int32)
            for k, (y, x) in enumerate(peaks, start=1):
                markers[y, x] = k
            labels_ws = watershed(-dist, markers, mask=binary)
            target = labels_ws[cy, cx]
            if target != 0:
                mask_ws = ((labels_ws == target) * 255).astype(np.uint8)
                ws_area = int((mask_ws > 0).sum())
                if ws_area >= 0.4 * median_area:
                    mask = mask_ws

    # ---- cap de raio ----
    if max_radius is not None:
        Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
        cap = ((X - cx) ** 2 + (Y - cy) ** 2) <= max_radius ** 2
        mask = (mask * cap).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask

# 32: crop = [45, 600, 243, 650],

def crop_seeds_centered(img, centroids, points, size_x, size_y, hiper=None,
                        columns_per_row=5, max_radius=None,
                        return_masks=True, mask_mode="auto", median_area=None):
    cropped_img, cropped_hiper, cropped_masks = [], [], []
    half_x, half_y = size_x // 2, size_y // 2

    if points is not None:
        points_ordered = ordenate_points(points, columns_per_row=columns_per_row)

    for i, cent in enumerate(centroids):
        x0, x1 = int(cent[0] - half_x), int(cent[0] + half_x)
        y0, y1 = int(cent[1] - half_y), int(cent[1] + half_y)

        crop = img[y0:y1, x0:x1].copy()
        label = points_ordered[i][1] if points is not None else ""

        # máscara leve, recalculada localmente
        local_cx = int(cent[0]) - x0
        local_cy = int(cent[1]) - y0
        mask = seed_mask_in_crop(crop, (local_cx, local_cy),
                                 max_radius=max_radius, mode=mask_mode,
                                 median_area=median_area)

        seed = crop
        if mask is not None:
            seed = cv2.bitwise_and(crop, crop, mask=mask)

        cropped_img.append([seed, label])
        cropped_masks.append(mask)

        if hiper is not None:
            hp = hiper[:, y0:y1, x0:x1].copy()
            if mask is not None:
                hp = hp * (mask[None, :, :].astype(np.float32) / 255.0)
            cropped_hiper.append(hp)

    if return_masks:
        return cropped_img, cropped_hiper, cropped_masks
    return cropped_img, cropped_hiper

def crop_seeds(img_rgb, centroids_rgb,
               img_rx, centroids_rx,
               hiper, points, size_x, size_y,):
  cropped_rgb,cropped_rx, cropped_hiper  = [], [], []
  half_x, half_y = size_x // 2, size_y // 2

  img_rx = cv2.cvtColor(img_rx, cv2.COLOR_BGR2GRAY)

  if points is not None:
    points_ordered = ordenate_points(points)

  for i, cent in enumerate(centroids_rgb):
    x0, x1 = int(cent[0] - half_x), int(cent[0] + half_x)
    y0, y1 = int(cent[1] - half_y), int(cent[1] + half_y)

    seed_rgb = img_rgb[y0:y1, x0:x1]
    hp = hiper[:, y0:y1, x0:x1]

    x0, x1 = int(centroids_rx[i][0] - half_x), int(centroids_rx[i][0] + half_x)
    y0, y1 = int(centroids_rx[i][1] - half_y), int(centroids_rx[i][1] + half_y)

    seed_rx = img_rx[y0:y1, x0:x1]
    _, thresh = cv2.threshold(seed_rx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    disk = morphology.disk(4)
    closed = morphology.closing(thresh, disk)

    converted = np.array(closed, dtype=int)
    mask = cv2.normalize(converted, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    if len(contours) > 1:
        biggest = [0,0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > biggest[0]:
                biggest = [area, cnt]

        cnt = biggest[1]

    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnt, -1, color=1, thickness=cv2.FILLED )
    mask_bool = mask.astype(bool)
    hull = morphology.convex_hull_image(mask_bool)
    mask = (hull).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    seed_rx = seed_rx * mask
    hp = hp * mask

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    seed_rgb = seed_rgb * mask

    label = points_ordered[i][1] if points is not None else ""

    seed_rgb = [seed_rgb, label]
    seed_rx = [seed_rx, label]
      
    cropped_rgb.append(seed_rgb)
    cropped_rx.append(seed_rx)
    cropped_hiper.append(hp)
  return cropped_rgb, cropped_rx, cropped_hiper