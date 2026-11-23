"""Entry point — runs all experiments described in the config."""
import argparse
import json

import torch, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig
from common.data import create_paths
from common.logger import ResultsLogger
from classification.evaluation import (
    evaluate_sklearn_models, evaluate_torch_models,
)
from classification.feature_extraction import Extractor
from classification.models import build_model


def main(config_path):
    with open(config_path) as f:
        cfg_json = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    logger = ResultsLogger(
        experiment_name=cfg_json["xai"]["experiment_name"],
        output_dir=cfg_json["paths"]["output"],
        config=cfg_json,
        seed=cfg_json["common"]["seed"],
    )

    tc = cfg_json.get("classification", {})
    train_cfg = TrainConfig(
        epochs=tc["torch_training"].get("epoch_num", 10),
        lr=tc["torch_training"].get("lr", 1e-3),
        weight_decay=tc["torch_training"].get("weight_decay", 8e-3),
        batch_size=tc["torch_training"].get("batch_size", 20),
        cv_folds=tc.get("cv_folds", 5),
        random_state=cfg_json["common"]["seed"],
        device=device,
    )

    path_data, labels = create_paths(cfg_json["paths"]["cropped_labeled"], is_labeled=True)

    print(f"Total de sementes anotadas: {len(path_data)}")
    print(f"Distribuição de classes: {dict(zip(*__import__('numpy').unique(labels, return_counts=True)))}")

    for model_name in cfg_json["classification"].get("models", []):
        print(f"\n\n########## Modelo: {model_name} ##########\n")
        bundle = build_model(model_name, device)

        extractor = Extractor(
            name=bundle.name,
            output_dir=cfg_json["classification"]["features_dir"],
            extractor=bundle.feature_extractor,
            preprocess=bundle.preprocess,
            device=device,
            
        )
        extractor.run(path_data, file_name="labeled")

        evaluate_sklearn_models(extractor, labels, train_cfg, output_path= cfg_json["paths"]["output"],
                                 model_name=bundle.name, logger=logger)
        evaluate_torch_models(extractor, labels, bundle.classifier_head,
                              bundle.classifier_index, train_cfg, output_path= cfg_json["paths"]["output"],
                              model_name=bundle.name, logger=logger)

     # Tabela cruzada (modelo x features) pronta pro artigo
    logger.finalize(
        also_latex=True,
        latex_pivot={
            "row_field": "model",
            "col_field": "features",
            "caption": "Acurácia média (± desvio) por modelo e conjunto de features",
            "label": "tab:classification_results",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)