"""Entry point — runs all experiments described in the config."""
import argparse
import json

import torch

from seed_classification.config import TorchTrainingConfig
from seed_classification.data import create_paths
from seed_classification.evaluation import (
    evaluate_sklearn_models, evaluate_torch_models,
)
from seed_classification.feature_extraction import Extractor
from seed_classification.models import build_model


def main(config_path):
    with open(config_path) as f:
        cfg_json = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tc = cfg_json.get("torch_training", {})
    train_cfg = TorchTrainingConfig(
        epoch_num=tc.get("epoch_num", 10),
        lr=tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 8e-3),
        batch_size=tc.get("batch_size", 20),
        cv_folds=cfg_json.get("cv_folds", 5),
        random_state=cfg_json.get("random_state", 42),
        device=device,
    )

    path_data, labels = create_paths(cfg_json["labeled_dir"], is_labeled=True)
    print(f"Total de sementes anotadas: {len(path_data)}")
    print(f"Distribuição de classes: {dict(zip(*__import__('numpy').unique(labels, return_counts=True)))}")

    for model_name in cfg_json.get("models", []):
        print(f"\n\n########## Modelo: {model_name} ##########\n")
        bundle = build_model(model_name, device)

        extractor = Extractor(
            name=bundle.name,
            output_dir=cfg_json["features_dir"],
            extractor=bundle.feature_extractor,
            preprocess=bundle.preprocess,
            device=device,
        )
        extractor.run(path_data)

        evaluate_sklearn_models(extractor, labels, train_cfg)
        evaluate_torch_models(
            extractor, labels, bundle.classifier_head,
            bundle.classifier_index, train_cfg,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)