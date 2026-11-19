import argparse
import json

import torch, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from common.data import PairedDataset, create_paths
from xai.io_utils import load_checkpoint
from xai.models import ThreeStreamResNet50, TwoStreamVGG19
from xai.transforms import val_transform
from xai.xai import coletar_exemplos_matriz, plotar_matriz_gradcam

# Nome dos streams por modelo (usado para puxar o caminho certo em dataset.dados)
STREAM_KEYS_2 = ("rgb", "rx")
STREAM_KEYS_3 = ("rgb", "rx", "hiper")


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg["xai"]["training"])
    set_seed(cfg["common"]["seed"])

    dados, rotulos = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)
    dataset = PairedDataset(dados, rotulos, val_transform)

    # --- VGG19 (2 streams) ---
    vgg = TwoStreamVGG19(num_classes=4).to(tc.device)
    vgg.load_state_dict(load_checkpoint(cfg["xai"]["checkpoints"]["vgg19"], tc.device))
    exemplos = coletar_exemplos_matriz(vgg, dataset, n_classes=4, device=tc.device)
    plotar_matriz_gradcam(vgg, dataset, exemplos, tc,
                          stream="rgb", grad=True, stream_keys=STREAM_KEYS_2)
    plotar_matriz_gradcam(vgg, dataset, exemplos, tc,
                          stream="rx", grad=True, stream_keys=STREAM_KEYS_2)

    # --- ResNet50 (3 streams) ---
    resnet = ThreeStreamResNet50(num_classes=4).to(tc.device)
    resnet.load_state_dict(
        load_checkpoint(cfg["xai"]["checkpoints"]["resnet50"], tc.device))
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