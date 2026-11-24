import argparse
import json
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.seed import set_seed
from common.data import create_paths
from xai.evaluation import evaluate_folds
from xai.models import ThreeStreamResNet50, TwoStreamVGG19, ThreeStreamInceptionV3
from xai.transforms import val_transform_inception


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tc = TrainConfig(**cfg["xai"]["training"])
    set_seed(cfg["common"]["seed"])

    dados, rotulos = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)
    base = os.path.join(cfg["paths"]["checkpoints"],
                        cfg["xai"]["experiment_name"])
    output_matrix = os.path.join(cfg["paths"]["output"], "conf_matrix_fine_tune")
    os.makedirs(output_matrix, exist_ok=True)

    evaluate_folds(
        dados, rotulos,
        build_model=lambda: TwoStreamVGG19(num_classes=4),
        streams = 2,
        checkpoint_pattern=os.path.join(base, "vgg19", "vgg19_fold_{fold}.pth"),
        cfg=tc,
        file_name= os.path.join(output_matrix, "matrix_vgg19")
    )
    evaluate_folds(
        dados, rotulos,
        build_model=lambda: ThreeStreamResNet50(num_classes=4),
        streams = 3,
        checkpoint_pattern=os.path.join(base, "resnet50", "resnet50_fold_{fold}.pth"),
        cfg=tc,
        file_name= os.path.join(output_matrix, "matrix_resnet50")
    )
    evaluate_folds(
        dados, rotulos,
        build_model=lambda: ThreeStreamInceptionV3(num_classes=4),
        streams = 3,
        checkpoint_pattern=os.path.join(base, "inception", "inception_fold_{fold}.pth"),
        cfg=tc,
        val_trans=val_transform_inception,
        file_name= os.path.join(output_matrix, "matrix_inception")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    main(parser.parse_args().config)