import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def split_merged_blob(mask, contour):
    local = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(local, [contour], -1, 255, thickness=-1)

    dist = ndi.distance_transform_edt(local)
    coords = peak_local_max(dist, min_distance=8, threshold_abs=4,
                            labels=local, exclude_border=False)
    if len(coords) < 2:
        return [contour]  # nada a separar

    markers = np.zeros_like(local, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k

    labels = watershed(-dist, markers, mask=local)
    out = []
    for k in range(1, labels.max() + 1):
        comp = ((labels == k) * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # pega o maior contorno da componente
            out.append(max(cnts, key=cv2.contourArea))
    return out

def _blob_stats(contours, mask):
    areas, solidities, peaks = [], [], []

    for c in contours:
        a = cv2.contourArea(c)
        areas.append(a)

        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        solidities.append(a / ha if ha > 0 else 0.0)

        local = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(local, [c], -1, 255, thickness=-1)
        dist = ndi.distance_transform_edt(local)
        pts = peak_local_max(dist, min_distance=5, threshold_abs=3,
                             labels=local, exclude_border=False)
        peaks.append(len(pts))

    return np.array(areas), np.array(solidities), np.array(peaks)


def _grid_slot_is_background(img, cx, cy, win=20):
    y0, y1 = max(0, cy - win), min(img.shape[0], cy + win)
    x0, x1 = max(0, cx - win), min(img.shape[1], cx + win)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, True

    if len(patch.shape) == 3:
        sat = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
        val = float(sat.mean())
    else:
        val = float(patch.mean())

    # fundo = saturação/brilho baixos. Ajuste os limiares conforme seu setup.
    return val, val < 40


def diagnose_tray(img, contours, centroids, mask, expected=50,
                  n_cols=5, verbose=False):
    result = {"total": len(contours), "expected": expected,
              "status": None, "merges": [], "missing_slots": [],
              "undetected_slots": []}

    if len(contours) == expected:
        result["status"] = "ok"
        if verbose:
            print(f"[diagnose] OK ({expected} sementes)")
        return result

    if len(contours) > expected:
        result["status"] = "excess"
        if verbose:
            print(f"[diagnose] EXCESSO: {len(contours)} > {expected} "
                  f"(provável split — erosão exagerada)")
        return result

    # ---- abaixo: len < expected ----
    areas, solid, peaks = _blob_stats(contours, mask)
    med = np.median(areas) if len(areas) else 1

    # sementes coladas: área >1.5x mediana OU solidez <0.85 OU ≥2 picos
    merge_idx = [i for i in range(len(contours))
                 if areas[i] > 1.5 * med
                 or solid[i] < 0.85
                 or peaks[i] >= 2]

    # slots vazios: reconstruir grade pelo centroide e checar gaps
    xs = np.array([c[0] for c in centroids])
    ys = np.array([c[1] for c in centroids])
    spacing = np.median(np.diff(np.sort(xs))) if len(xs) > 1 else 50
    spacing_y = np.median(np.diff(np.sort(ys))) if len(ys) > 1 else 50
    result["spacing_x"] = float(spacing)
    result["spacing_y"] = float(spacing_y)
    # posições esperadas da grade: 5 col × 10 linhas
    # estima x0/y0 pelo mínimo, e gera grade teórica
    x0, y0 = xs.min(), ys.min()
    expected_pts = [(x0 + c * spacing, y0 + r * spacing_y)
                    for r in range(expected // n_cols)
                    for c in range(n_cols)]

    for (ex, ey) in expected_pts:
        # algum centróide próximo?
        d = np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2)
        if d.min() > 0.5 * min(spacing, spacing_y):
            val, is_bg = _grid_slot_is_background(
                img, int(ex), int(ey),
            )
            if is_bg:
                result["missing_slots"].append((int(ex), int(ey)))
            else:
                result["undetected_slots"].append((int(ex), int(ey)))

    result["merges"] = merge_idx

    deficit = expected - len(contours)
    has_merge = len(merge_idx) >= deficit
    has_missing = len(result["missing_slots"]) >= deficit
    has_undetected = len(result["undetected_slots"]) >= deficit

    if has_merge and not has_missing and not has_undetected:
        result["status"] = "merged"
    elif has_missing and not has_merge and not has_undetected:
        result["status"] = "missing"
    elif has_undetected and not has_merge and not has_missing:
        result["status"] = "undetected"
    else:
        result["status"] = "mixed"
    

    if verbose:
        print(f"[diagnose] {len(contours)}/{expected} → status={result['status']}")
        print(f"  merges suspeitos : {result['merges']}")
        print(f"  slots vazios(fundo): {result['missing_slots']}")
        print(f"  slots vazios(sinal): {result['undetected_slots']}")
    return result