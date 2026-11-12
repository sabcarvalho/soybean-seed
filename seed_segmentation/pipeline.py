"""Main segmentation pipeline."""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .segmentation import segmentation_rgb, segmentation_rx
from .geometry import calculate_centroids, ordenate_points, homography
from .cropping import crop_seeds_centered
from .io_utils import anote_points


class SegmentationPipeline:
    def __init__(self, img, cfg):
        self.rgb_img = img["rgb"]
        self.rx_img = img["rx"]
        self.hiper_imgs = img["hiper"]
        self.json = img["json"]
        self.cfg = cfg
        self.SIZEX, self.SIZEY = cfg.crop_size

    def run(self):
        points = anote_points(self.json) if self.json is not None else None
        cols = self.cfg.columns_per_row

        # RGB
        contours = segmentation_rgb(self.rgb_img, self.cfg, self.cfg.roi)
        centroids = calculate_centroids(contours)
        ord_cen_rgb, _ = ordenate_points(centroids, contours, cols)

        # RX
        contours_rx = segmentation_rx(self.rx_img, self.cfg, rotate=True)
        centroids_rx = calculate_centroids(contours_rx)
        ord_cen_rx, _ = ordenate_points(centroids_rx, contours_rx, cols)

        # Homography RX → RGB
        new_rx = homography(self.rx_img, ord_cen_rx, self.rgb_img,
                            ord_cen_rgb, self.cfg.ransac_threshold)

        new_cnt_rx = segmentation_rx(new_rx, self.cfg, rotate=False)
        new_cen_rx = calculate_centroids(new_cnt_rx)
        ocen, _ = ordenate_points(new_cen_rx, new_cnt_rx, cols)

        # Crops
        self.cropped_rgb, self.cropped_hiper = crop_seeds_centered(
            self.rgb_img, ord_cen_rgb, points,
            self.SIZEX, self.SIZEY, hiper=self.hiper_imgs, columns_per_row=cols)

        self.cropped_rx, _ = crop_seeds_centered(
            new_rx, ocen, points, self.SIZEX, self.SIZEY,
            columns_per_row=cols)

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