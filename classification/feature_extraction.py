"""Feature extraction from RGB, RX and hyperspectral cubes."""
import os

import cv2
import numpy as np
import torch
from PIL import Image


class Extractor:
    def __init__(self, name, output_dir, extractor=None, preprocess=None,
                 rgb_only=False, device=None):
        self.extractor = extractor
        self.preprocess = preprocess
        self.path = os.path.join(output_dir, name)
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if extractor is not None:
            self.extractor = self.extractor.to(self.device)
            self.extractor.eval()
        self.data = None
        self.rgb_only = rgb_only
        self.len_hiper = None

    def extract_visual_features(self, image_np):
        img = Image.fromarray(image_np)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.extractor(tensor)
        return feat.flatten().cpu().numpy()

    def extract_multimodal_seed_features(self, img_rgb, img_rx, cube_hiper):
        feat_rgb = self.extract_visual_features(img_rgb)
        if self.rgb_only:
            return feat_rgb

        if len(img_rx.shape) == 2:
            img_rx_3ch = cv2.cvtColor(img_rx, cv2.COLOR_GRAY2RGB)
        else:
            img_rx_3ch = img_rx
        feat_rx = self.extract_visual_features(img_rx_3ch)

        feat_hiper = np.mean(cube_hiper, axis=(1, 2))
        self.len_hiper = feat_hiper.shape[0]

        return np.concatenate((feat_rgb, feat_rx, feat_hiper))

    def load_and_extract_matrix(self, path_list):
        X = []
        for i, paths in enumerate(path_list):
            img_bgr = cv2.imread(paths["rgb"])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rx = cv2.imread(paths["rx"], cv2.IMREAD_GRAYSCALE)
            cube_hiper = np.load(paths["hiper"])

            X.append(self.extract_multimodal_seed_features(img_rgb, img_rx, cube_hiper))
            if (i + 1) % 100 == 0:
                print(f"  Processadas {i + 1}/{len(path_list)} sementes...")
        return np.array(X)

    def run(self, path_list, file_name="data"):
        os.makedirs(self.path, exist_ok=True)
        data_path = os.path.join(self.path, f"{file_name}.npy")

        if os.path.exists(data_path):
            self.data = np.load(data_path)
            if self.len_hiper is None:
                self.len_hiper = 150
        else:
            if self.extractor is None:
                print("Erro: Extrator nulo, sem resultados guardados!")
                return
            print("-> Extraindo características...")
            self.data = self.load_and_extract_matrix(path_list)
            np.save(data_path, self.data)

        print(f"Caracteristicas salvas com sucesso em: {self.path}")

    def slice_data(self):
        len_rgb = int(self.data.shape[1] - self.len_hiper) // 2
        rgb = self.data[:, :len_rgb]
        rx = self.data[:, len_rgb:2 * len_rgb]
        hiper = self.data[:, 2 * len_rgb:]
        return rgb, rx, hiper