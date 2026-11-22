"""Main segmentation pipeline."""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .segmentation import segmentation_rgb, segmentation_rx, retry_segment_slot
from .geometry import calculate_centroids, ordenate_points, homography
from .cropping import crop_seeds_centered, crop_seeds
from .io_utils import anote_points
from .diagnostic import diagnose_tray, split_merged_blob


class SegmentationPipeline:
    def __init__(self, img, cfg):
        self.rgb_img = img["rgb"]
        self.rx_img = img["rx"]
        self.hiper_imgs = img["hiper"]
        self.json = img["json"]
        self.cfg = cfg
        self.SIZEX, self.SIZEY = cfg.crop_size

    def _test_tray(self,contours, centroids, rgb_mask, cols, expected):
        diag = diagnose_tray(self.rgb_img, contours, centroids, rgb_mask,
                                    expected=expected, n_cols=cols, verbose=False)
        
        if diag["status"] in ("merged", "mixed") and diag["merges"]:
            new_contours = []
            for i, c in enumerate(contours):
                if i in diag["merges"]:
                    new_contours.extend(split_merged_blob(rgb_mask, c))
                else:
                    new_contours.append(c)
            contours = tuple(new_contours)
            centroids = calculate_centroids(contours)

        if diag["status"] in ("undetected", "mixed") and diag["undetected_slots"]:
            # ROI um pouco menor que o espaçamento entre sementes
            roi_size = int(min(diag["spacing_x"], diag["spacing_y"]) * 0.9)
            new_contours = list(contours)
            for (cx, cy) in diag["undetected_slots"]:
                c = retry_segment_slot(
                    self.rgb_img, cx, cy,
                    mode="rgb",
                    roi_size=roi_size,
                    threshold_factor=0.6,
                    erosion=1,
                    max_offset=roi_size // 3,
                    verbose=True,
                )
                if c is not None:
                    new_contours.append(c)
            contours = tuple(new_contours)
            centroids = calculate_centroids(contours)
        diag_final = diagnose_tray(self.rgb_img, contours, centroids, rgb_mask,
                                    expected=expected, n_cols=cols, verbose=False)
        self.diag = diag_final
        return contours, centroids

    def run(self):
        points = anote_points(self.json) if self.json is not None else None
        cols = self.cfg.columns_per_row

        contours, rgb_mask = segmentation_rgb(
            self.rgb_img, self.cfg, return_mask=True
        )
        centroids = calculate_centroids(contours)
        ord_cen_rgb, _ = ordenate_points(centroids, contours, cols)

        expected = self.cfg.expected_seeds
        contours, centroids = self._test_tray(contours, centroids, rgb_mask, cols, expected)
        
        if self.diag["status"] != "ok":
            raise ValueError(
                f"Houve divergencia na qtde de sementes encontradas em relacao as esperadas: {len(contours)}/{expected}. Abortando..."
            )

        ord_cen_rgb, _ = ordenate_points(centroids, contours, cols)

        contours_rx = segmentation_rx(
            self.rx_img, self.cfg, rotate=True
        )
        centroids_rx = calculate_centroids(contours_rx)
        ord_cen_rx, _ = ordenate_points(centroids_rx, contours_rx, cols)

        new_rx = homography(self.rx_img, ord_cen_rx, self.rgb_img,
                            ord_cen_rgb, self.cfg.ransac_threshold)

        new_cnt_rx = segmentation_rx(new_rx, self.cfg, rotate=False)
        new_cen_rx = calculate_centroids(new_cnt_rx)
        ocen, _ = ordenate_points(new_cen_rx, new_cnt_rx, cols)
        self.cropped_rgb, self.cropped_rx, self.cropped_hiper = crop_seeds(self.rgb_img, ord_cen_rgb, 
                                                                      new_rx, ocen, self.hiper_imgs, points, 
                                                                      self.SIZEX, self.SIZEY, self.cfg.disk_open, self.cfg.disk_close)

    def save_cropped_seeds(self, output_dir, image_id):
        rgb_dir = os.path.join(output_dir, "RGB")
        rx_dir = os.path.join(output_dir, "RX")
        hiper_dir = os.path.join(output_dir, "HIPER")
        for d in (rgb_dir, rx_dir, hiper_dir):
            os.makedirs(d, exist_ok=True)

        for n, seed in enumerate(self.cropped_rgb):
            name = f"{image_id}_seed_{n:02d}_{seed[1]}.png"
            if isinstance(seed[0], np.ndarray):
                cv2.imwrite(os.path.join(rgb_dir, name), seed[0])

            rx = self.cropped_rx[n][0]
            if isinstance(rx, np.ndarray):
                cv2.imwrite(os.path.join(rx_dir, name), rx)

            if self.cropped_hiper is not None and n < len(self.cropped_hiper):
                np.save(
                    os.path.join(hiper_dir, f"{image_id}_seed_{n:02d}.npy"),
                    self.cropped_hiper[n],
                )
        print(f"-> {len(self.cropped_rgb)} sementes da {image_id} salvas em {output_dir}")

    def plot_seed(self, n_seed, n_band):
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        rgb = cv2.cvtColor(self.cropped_rgb[n_seed][0], cv2.COLOR_BGR2RGB)

        axs[0].imshow(rgb); axs[0].set_title("RGB"); axs[0].axis("off")
        axs[1].imshow(self.cropped_rx[n_seed]); axs[1].set_title("RX"); axs[1].axis("off")
        axs[2].imshow(self.cropped_hiper[n_seed][n_band])
        axs[2].set_title(f"Hiper — banda {n_band}"); axs[2].axis("off")

        blended = cv2.addWeighted(rgb, 0.5, self.cropped_rx[n_seed], 0.5, 0)
        axs[3].imshow(blended); axs[3].set_title("RGB + RX"); axs[3].axis("off")
        plt.tight_layout(); plt.show()