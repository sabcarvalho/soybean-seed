"""Configuration for the seed segmentation pipeline."""
from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    roi_rgb: tuple = (30, 600, 200, 650)      # y0, y1, x0, x1
    roi_rx: tuple = (109, 2331, 0, 2368)      # y0, y1, x0, x1
    crop_size: tuple = (64, 45)           # (SIZEX, SIZEY)
    columns_per_row: int = 5
    n_hyperspectral_bands: int = 150
    expected_seeds: int = 50

    # Morphology radii
    rgb_erosion: int = 13
    rgb_dilation: int = 3
    rx_erosion: int = 5
    rx_dilation: int = 2

    # Contour filters
    min_circularity: float = 0.3
    min_area: float = 0.3

    disk_open: int = 2
    disk_close: int = 1

    # Homography
    ransac_threshold: float = 1.0


@dataclass
class FileTemplates:
    rgb: str = "RIP2_{id}-RGB.tif"
    rx: str = "RIP2_{id}.bmp"
    bil: str = "RIP2_{id}.bil"
    hdr: str = "RIP2_{id}.bil.hdr"
    json: str = "RIP2_{id}-RGB.json"