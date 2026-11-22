"""Roda LabelSpreading/Propagation nos crops não anotados."""
import argparse
import json
import torch, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.config import TrainConfig, SSLConfig
from common.data import create_paths
from common.logger import ResultsLogger
from common.seed import set_seed
from classification.models import MODEL_BUILDERS
from classification.feature_extraction import Extractor
from semi_supervised.label_spreading import evaluate_graph_ssl
from semi_supervised.pseudo_labeling import evaluate_sklearn_ssl



def run_features(cfg, ssl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled, y = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)
    unlabeled, _ = create_paths(cfg["paths"]["cropped_unlabeled"], is_labeled=False)

    logger = ResultsLogger("ssl_features", cfg["paths"]["output"],
                           config=cfg, seed=cfg["common"]["seed"])

    for name in cfg["classification"]["models"]:
        bundle = MODEL_BUILDERS[name](device)
        ext = Extractor(bundle.name, cfg["classification"]["features_dir"],
                        extractor=bundle.feature_extractor,
                        preprocess=bundle.preprocess, device=device)
        ext.run(labeled, file_name="labeled")
        ext_unl = Extractor(bundle.name, cfg["classification"]["features_dir"],
                            extractor=bundle.feature_extractor,
                            preprocess=bundle.preprocess, device=device)
        ext_unl.run(unlabeled, file_name="unlabeled")

        evaluate_sklearn_ssl(ext, y, ext_unl, ssl_cfg,
                             model_name=bundle.name, logger=logger)

    logger.finalize()



def run_label_spreading(cfg):
    set_seed(cfg["common"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled, y = create_paths(cfg["paths"]["cropped_labeled"], is_labeled=True)
    unlabeled, _ = create_paths(cfg["paths"]["cropped_unlabeled"],
                              is_labeled=False)

    logger = ResultsLogger("graph_ssl", cfg["paths"]["output"],
                           config=cfg, seed=cfg["common"]["seed"])

    for name in cfg["classification"]["models"]:
        bundle = MODEL_BUILDERS[name](device)

        ext_l = Extractor(bundle.name, cfg["classification"]["features_dir"],
                          extractor=bundle.feature_extractor,
                          preprocess=bundle.preprocess, device=device)
        ext_l.run(labeled, file_name="labeled")

        ext_u = Extractor(bundle.name, cfg["classification"]["features_dir"],
                          extractor=bundle.feature_extractor,
                          preprocess=bundle.preprocess, device=device)
        ext_u.run(unlabeled, file_name="unlabeled")

        evaluate_graph_ssl(ext_l, y, ext_u, cfg["classification"],
                           model_name=bundle.name, logger=logger)

    logger.finalize()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    set_seed(cfg["common"]["seed"])

    ssl_cfg = SSLConfig(**cfg.get("semi_supervised", {}))
    tc = TrainConfig(**cfg.get("xai", {}).get("training", {}))

    run_label_spreading(cfg)
    run_features(cfg, ssl_cfg)