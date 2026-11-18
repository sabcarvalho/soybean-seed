import argparse
import json

import torch, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from seed_xai.config import (STREAM_KEYS_2, STREAM_KEYS_3)
from common.data import PairedDataset, create_paths
from seed_xai.io_utils import load_checkpoint
from seed_xai.models import ThreeStreamResNet50, TwoStreamVGG19
from seed_xai.transforms import val_transform
from seed_xai.xai import coletar_exemplos_matriz, plotar_matriz_gradcam


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg.get("training", {}))
    set_seed(cfg.get("seed", 42))

    dados, rotulos = create_paths(cfg["labeled_dir"], is_labeled=True)
    dataset = PairedDataset(dados, rotulos, val_transform)

    # --- VGG19 (2 streams) ---
    vgg = TwoStreamVGG19(num_classes=4).to(tc.device)
    vgg.load_state_dict(load_checkpoint(cfg["xai"]["vgg_checkpoint"], tc.device))
    exemplos = coletar_exemplos_matriz(vgg, dataset, n_classes=4, device=tc.device)
    plotar_matriz_gradcam(vgg, dataset, exemplos, tc,
                          stream="rgb", grad=True, stream_keys=STREAM_KEYS_2)
    plotar_matriz_gradcam(vgg, dataset, exemplos, tc,
                          stream="rx", grad=True, stream_keys=STREAM_KEYS_2)

    # --- ResNet50 (3 streams) ---
    resnet = ThreeStreamResNet50(num_classes=4).to(tc.device)
    resnet.load_state_dict(
        load_checkpoint(cfg["xai"]["resnet_checkpoint"], tc.device))
    exemplos = coletar_exemplos_matriz(resnet, dataset, n_classes=4, device=tc.device)
    plotar_matriz_gradcam(resnet, dataset, exemplos, tc,
                          stream="rgb", grad=True, stream_keys=STREAM_KEYS_3)
    plotar_matriz_gradcam(resnet, dataset, exemplos, tc,
                          stream="rx", grad=True, stream_keys=STREAM_KEYS_3)
    # controle: mesma matriz sem Grad-CAM
    plotar_matriz_gradcam(resnet, dataset, exemplos, tc,
                          stream="rgb", grad=False, stream_keys=STREAM_KEYS_3)
    plotar_matriz_gradcam(resnet, dataset, exemplos, tc,
                          stream="rx", grad=False, stream_keys=STREAM_KEYS_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)