"""Dataloaders e descoberta de paths."""
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(self, dados, labels, transform, streams = 3):
        self.dados = dados
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.streams = streams

    def __len__(self):
        return len(self.dados)

    def _load_hiper(self, path):
        cube = np.load(path)                     # (bands, h, w)
        return torch.tensor(np.mean(cube, axis=(1, 2)), dtype=torch.float32)

    def __getitem__(self, idx):
        caminhos = self.dados[idx]

        img_bgr = cv2.imread(caminhos["rgb"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_rx = cv2.imread(caminhos["rx"], cv2.IMREAD_GRAYSCALE)
        img_rx_3ch = cv2.cvtColor(img_rx, cv2.COLOR_GRAY2RGB)

        pil_rgb = Image.fromarray(img_rgb)
        pil_rx = Image.fromarray(img_rx_3ch)

        # Mesma augmentation (mesma seed) entre os dois streams
        seed = torch.randint(0, 2 ** 32, size=(1,)).item()
        torch.manual_seed(seed)
        tensor_rgb = self.transform(pil_rgb)
        torch.manual_seed(seed)
        tensor_rx = self.transform(pil_rx)

        # 3º stream opcional: se o caminho existe e o .npy foi carregado
        if self.streams == 3:
            tensor_hiper = self._load_hiper(caminhos["hiper"])
            return tensor_rgb, tensor_rx, tensor_hiper, self.labels[idx]

        return tensor_rgb, tensor_rx, self.labels[idx]