import argparse
import json, torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import TrainConfig
from common.seed import set_seed
from common.data import create_paths
from classification.models import build_inception_v3
from classification.data import CustomDataset
from classification.feature_extraction import Extractor
from xai.data import PairedDataset
from xai.io_utils import load_checkpoint
from xai.models import ThreeStreamResNet50, TwoStreamVGG19, ThreeStreamInceptionV3
from xai.transforms import val_transform
from xai.xai import coletar_exemplos_matriz, plotar_matriz_gradcam
from xai.transforms import (
    val_transform, val_transform_inception
)
# Nome dos streams por modelo (usado para puxar o caminho certo em dataset.dados)
STREAM_KEYS_2 = ("rgb", "rx")
STREAM_KEYS_3 = ("rgb", "rx", "hiper")


def _run_matrices(model, dataset, stream_keys, img_size, cfg,
                  streams=("rgb", "rx"), with_control=False,
                  tag="model"):
    """Gera as matrizes (com e sem Grad-CAM) para um modelo."""
    exemplos = coletar_exemplos_matriz(
        model, dataset, n_classes=4, device=cfg.device
    )
    print(f"  [{tag}] {len(exemplos)} quadrantes com exemplo")

    for stream in streams:
        plotar_matriz_gradcam(
            model, dataset, exemplos, cfg,
            stream=stream, grad=True,
            stream_keys=stream_keys,
            img_size=img_size,
        )
        if with_control:
            plotar_matriz_gradcam(
                model, dataset, exemplos, cfg,
                stream=stream, grad=False,
                stream_keys=stream_keys,
                img_size=img_size,
            )


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg.get("training", {}))
    set_seed(cfg.get("seed", 42))

    dados, rotulos = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)

    # ══════════════════════════════════════════════════════════════════
    # VGG19 — 2 streams, input 224
    # ══════════════════════════════════════════════════════════════════
    ds_vgg = PairedDataset(dados, rotulos, val_transform)
    vgg = TwoStreamVGG19(num_classes=4).to(tc.device)
    vgg.load_state_dict(
        load_checkpoint(cfg["xai"]["checkpoints"]["vgg19"], tc.device)
    )
    _run_matrices(vgg, ds_vgg,
                  stream_keys=STREAM_KEYS_2,
                  img_size=224, cfg=tc, tag="vgg19")

    # ══════════════════════════════════════════════════════════════════
    # ResNet50 — 3 streams, input 224
    # ══════════════════════════════════════════════════════════════════
    ds_resnet = PairedDataset(dados, rotulos, val_transform)
    resnet = ThreeStreamResNet50(num_classes=4).to(tc.device)
    resnet.load_state_dict(
        load_checkpoint(cfg["xai"]["checkpoints"]["resnet50"], tc.device)
    )
    _run_matrices(resnet, ds_resnet,
                  stream_keys=STREAM_KEYS_3,
                  img_size=224, cfg=tc, with_control=True,
                  tag="resnet50")

    # ══════════════════════════════════════════════════════════════════
    # InceptionV3 
    # ══════════════════════════════════════════════════════════════════
    inception = build_inception_v3(tc.device)
    extractor = Extractor(
        name=inception.name,
        output_dir=cfg["classification"]["features_dir"],
        extractor=inception.feature_extractor,
        preprocess=inception.preprocess,
        device=tc.device,
        
    )
    inception_classifier = inception.classifier_head
    extractor.run(dados, file_name="labeled")
    ds_inception = CustomDataset(extractor.data, rotulos)
    inception_classifier[inception.classifier_index] = torch.nn.Linear(
        in_features=extractor.data.shape[1], out_features=4, bias=True
    )
    inception_classifier.load_state_dict(
        load_checkpoint(cfg["xai"]["checkpoints"]["inception_v3"], tc.device)
    )
    inception_classifier.to(tc.device)
    _run_matrices(inception_classifier, ds_inception,
                  stream_keys=STREAM_KEYS_3,
                  img_size=299, cfg=tc, with_control=True,
                  tag="inception")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)