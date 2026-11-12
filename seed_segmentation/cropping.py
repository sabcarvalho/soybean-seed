"""Crop individual seeds centered on centroids."""
from .geometry import ordenate_points


def crop_seeds_centered(img, centroids, points, size_x, size_y,
                        hiper=None, columns_per_row=5):
    cropped_img, cropped_hiper = [], []
    half_x, half_y = size_x // 2, size_y // 2

    if points is not None:
        points_ordered = ordenate_points(points, columns_per_row=columns_per_row)

    for i, cent in enumerate(centroids):
        x0, x1 = int(cent[0] - half_x), int(cent[0] + half_x)
        y0, y1 = int(cent[1] - half_y), int(cent[1] + half_y)

        seed = img[y0:y1, x0:x1]
        label = points_ordered[i][1] if points is not None else ""
        cropped_img.append([seed, label])

        if hiper is not None:
            cropped_hiper.append(hiper[:, y0:y1, x0:x1])
    return cropped_img, cropped_hiper