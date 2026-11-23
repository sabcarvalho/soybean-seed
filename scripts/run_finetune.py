import argparse
import json
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from common.logger import ResultsLogger
from common.data import create_paths
from xai.models import ThreeStreamResNet50, TwoStreamVGG19, ThreeStreamInceptionV3
from xai.training import cross_validate_finetune
from xai.transforms import (
    train_transform, val_transform,
    train_transform_inception, val_transform_inception,
)

MODEL_BUILDERS = {
    "vgg19": (lambda: TwoStreamVGG19(num_classes=4),
              train_transform, val_transform, 2),
    "resnet50": (lambda: ThreeStreamResNet50(num_classes=4),
                 train_transform, val_transform, 3),
    "inception": (lambda: ThreeStreamInceptionV3(num_classes=4),
                train_transform_inception, val_transform_inception),
}


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg["xai"]["training"])
    set_seed(cfg["common"]["seed"])

    logger = ResultsLogger(
        experiment_name=cfg["xai"]["experiment_name"],
        output_dir=cfg["paths"]["output"],
        config=cfg, seed=cfg.get("seed", 42),
    )

    dados, rotulos = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)
    print(f"Total de sementes: {len(dados)}")

    for model_name, (build, train_tf, val_tf, streams) in MODEL_BUILDERS.items():
        out_dir = os.path.join(
            cfg["paths"]["checkpoints"],
            cfg["xai"]["experiment_name"],
            model_name,
        )
        print(f"\n### {model_name} → {out_dir}")
        cross_validate_finetune(dados, rotulos, build, streams, tc,
                                 checkpoint_dir=out_dir,
                                 model_tag=model_name,
                                 train_tf=train_tf,
                                 val_tf=val_tf,
                                 logger=logger)
    logger.finalize(
        also_latex=True,
        latex_pivot={
            "row_field": "model",
            "col_field": "fold",
            "mean_field": "final_val_acc",
            "std_field": "fold_duration_s",  # ou omitir
            "caption": "Acurácia de validação por fold",
            "label": "tab:finetune_folds",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)