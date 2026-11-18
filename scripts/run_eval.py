import argparse
import json
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from common.data import create_paths
from seed_xai.evaluation import evaluate_folds
from seed_xai.models import ThreeStreamResNet50, TwoStreamVGG19


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg.get("fine_tuning", {}))
    set_seed(cfg.get("seed", 42))

    dados, rotulos = create_paths(cfg["labeled_dir"], is_labeled=True)
    base = os.path.join(cfg["checkpoint_dir"],
                        cfg.get("experiment_name", "exp1"))

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