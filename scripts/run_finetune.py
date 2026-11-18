import argparse
import json
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from common.data import create_paths
from seed_xai.models import ThreeStreamResNet50, TwoStreamVGG19
from seed_xai.training import cross_validate_finetune


MODEL_BUILDERS = {
    "vgg19": lambda: TwoStreamVGG19(num_classes=4),
    "resnet50": lambda: ThreeStreamResNet50(num_classes=4),
}


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg.get("fine_tuning", {}))
    set_seed(cfg.get("seed", 42))

    dados, rotulos = create_paths(cfg["labeled_dir"], is_labeled=True)
    print(f"Total de sementes: {len(dados)}")

    for model_name, build in MODEL_BUILDERS.items():
        out_dir = os.path.join(
            cfg["checkpoint_dir"],
            cfg.get("experiment_name", "exp1"),
            model_name,
        )
        print(f"\n### {model_name} → {out_dir}")
        cross_validate_finetune(dados, rotulos, build, tc,
                                checkpoint_dir=out_dir,
                                model_tag=model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)