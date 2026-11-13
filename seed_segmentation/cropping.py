import cv2
import numpy as np
from .geometry import ordenate_points

def seed_mask_in_crop(crop, center_local, max_radius=None, mode="auto"):
    if crop is None or crop.size == 0:
        return None

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

    _, bin_img = cv2.threshold(feature, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if mode == "rx" and bin_img[cy, cx] == 0:
        bin_img = cv2.bitwise_not(bin_img)

    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if bin_img[cy, cx] == 0:
        cv2.circle(bin_img, (cx, cy), 2, 255, -1)

    _, labels = cv2.connectedComponents(bin_img)
    lbl = labels[cy, cx]
    if lbl == 0:
        return None
    mask = ((labels == lbl) * 255).astype(np.uint8)

    if max_radius is not None:
        Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
        cap = ((X - cx) ** 2 + (Y - cy) ** 2) <= max_radius ** 2
        mask = (mask * cap).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask



def crop_seeds_centered(img, centroids, points, size_x, size_y, hiper=None,
                        columns_per_row=5, max_radius=None,
                        return_masks=True, mask_mode="auto"):
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
                                 max_radius=max_radius, mode=mask_mode)

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