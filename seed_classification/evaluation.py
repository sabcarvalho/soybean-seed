"""High-level experiment drivers."""
import copy

import numpy as np
import torch

from .classifiers import RandomForest, SupportVectorMachine, XGBoost
from .torch_training import do_cross_val_torch, reset_weights

def _run_and_log(classifiers, data, labels, feature_name, model_name,
                 cfg, logger):
    title = f"==============Treinando com {feature_name}============="
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    for name, clf in classifiers:
        print(name)
        scores = clf.cross_val(data, labels, cv_folds=cfg.cv_folds,
                               random_state=cfg.random_state)
        if logger is not None:
            row = {
                "model": model_name,
                "classifier": name,
                "features": feature_name,
                "n_samples": len(labels),
                "mean_acc": float(scores.mean()),
                "std_acc": float(scores.std()),
            }
            for i, s in enumerate(scores):
                row[f"fold_{i + 1}_acc"] = float(s)
            logger.add(**row)

def evaluate_sklearn_models(data_extracted, labels, cfg,
                             model_name="unknown", logger=None):
    classifiers = [
        ("SVM", SupportVectorMachine()),
        ("RandomForest", RandomForest(random_state=cfg.random_state)),
        ("XGBoost", XGBoost(n_classes=len(set(labels)),
                            random_state=cfg.random_state)),
    ]
    _run_and_log(classifiers, data_extracted.data, labels, "all", model_name, cfg, logger)
    rgb, rx, hiper = data_extracted.slice_data()
    _run_and_log(classifiers, rgb, labels, "RGB", model_name, cfg, logger)
    _run_and_log(classifiers, rx,   labels, "RX", model_name, cfg, logger)
    _run_and_log(classifiers, hiper,labels, "HIPER", model_name, cfg, logger)
    _run_and_log(classifiers, np.concatenate((rgb, rx), axis=1),    labels, "RGB+RX", model_name, cfg, logger)
    _run_and_log(classifiers, np.concatenate((rgb, hiper), axis=1), labels, "RGB+HIPER", model_name, cfg, logger)
    _run_and_log(classifiers, np.concatenate((rx, hiper), axis=1),  labels, "RX+HIPER", model_name, cfg, logger)



def _run_torch_and_log(classifier, data, labels, feature_name,
                       model_name, cfg, logger):
    title = f"==============Treinando com {title}============="
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    classifier.apply(reset_weights)
    accs, losses = do_cross_val_torch(data, labels, classifier, cfg)
    if logger is not None:
        row = {
            "model": model_name,
            "classifier": "torch_head",
            "features": feature_name,
            "n_samples": len(labels),
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
        }
        for i, a in enumerate(accs):
            row[f"fold_{i + 1}_acc"] = float(a)
        logger.add(**row)

def evaluate_torch_models(data_extracted, labels, head, classifier_index, cfg,
                          model_name, logger):
    head = copy.deepcopy(head)
    out_feat = head[classifier_index].out_features

    def _set_in_features(n):
        head[classifier_index] = torch.nn.Linear(
            in_features=n, out_features=out_feat, bias=True
        )

    _set_in_features(data_extracted.data.shape[1])
    _run_torch_and_log(head, data_extracted.data, labels, "all", model_name, cfg, logger)

    rgb, rx, hiper = data_extracted.slice_data()
    _set_in_features(rgb.shape[1])
    _run_torch_and_log(head, rgb, labels, "RGB", model_name, cfg, logger)

    _set_in_features(rx.shape[1]) 
    _run_torch_and_log(head, rx, labels, "RX", model_name, cfg, logger)

    _set_in_features(hiper.shape[1])
    _run_torch_and_log(head, hiper, labels, "HIPER", model_name, cfg, logger)

    _set_in_features(np.concatenate((rgb, rx), axis=1).shape[1])
    _run_torch_and_log(head, np.concatenate((rgb, rx), axis=1), labels, "RGB+RX", model_name, cfg, logger)
    
    _set_in_features(np.concatenate((rgb, hiper), axis=1).shape[1])
    _run_torch_and_log(head, np.concatenate((rgb, hiper), axis=1), labels, "RGB+HIPER", model_name, cfg, logger)
    
    _set_in_features(np.concatenate((rx, hiper), axis=1).shape[1])
    _run_torch_and_log(head, np.concatenate((rx, hiper), axis=1), labels, "RX+HIPER", model_name, cfg, logger)
    