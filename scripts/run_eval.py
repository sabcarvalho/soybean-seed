import argparse
import json
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from common.data import create_paths
from xai.evaluation import evaluate_folds
from xai.models import ThreeStreamResNet50, TwoStreamVGG19


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg["xai"]["training"])
    set_seed(cfg["common"]["seed"])

    dados, rotulos = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)
    base = os.path.join(cfg["paths"]["checkpoint"],
                        cfg["xai"]["experiment_name"])

    evaluate_folds(
        dados, rotulos,
        build_model=lambda: TwoStreamVGG19(num_classes=4),
        checkpoint_pattern=os.path.join(base, "vgg19", "vgg19_fold_{fold}.pth"),
        cfg=tc,
    )
    evaluate_folds(
        dados, rotulos,
        build_model=lambda: ThreeStreamResNet50(num_classes=4),
        checkpoint_pattern=os.path.join(base, "resnet50", "resnet50_fold_{fold}.pth"),
        cfg=tc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)