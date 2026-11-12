"""I/O helpers for reading seeds and annotations."""
import json
import os
import cv2
import numpy as np
from spectral import envi


def anote_points(data):
    """Extract [point, label] from a labelme-style JSON."""
    return [[shape["points"][0], shape["label"]] for shape in data["shapes"]]


def read_seed_entry(image_id, data_dir, templates, n_bands=150,
                    load_json=True):
    """Load all modalities for a given image id."""
    rgb = cv2.imread(os.path.join(data_dir, templates.rgb.format(id=image_id)))
    rx = cv2.imread(os.path.join(data_dir, templates.rx.format(id=image_id)))

    hdr = os.path.join(data_dir, templates.hdr.format(id=image_id))
    bil = os.path.join(data_dir, templates.bil.format(id=image_id))
    cube = envi.open(hdr, bil)
    hiper = np.array([cube.read_band(i) for i in range(n_bands)])

    json_data = None
    if load_json:
        with open(os.path.join(data_dir, templates.json.format(id=image_id))) as f:
            json_data = json.load(f)

    return {"rgb": rgb, "rx": rx, "hiper": hiper, "json": json_data}