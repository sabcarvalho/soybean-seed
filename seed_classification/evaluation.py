"""High-level experiment drivers."""
import copy

import numpy as np
import torch

from .classifiers import RandomForest, SupportVectorMachine, XGBoost
from .torch_training import do_cross_val_torch, reset_weights


def _train_sklearn(classifiers, data, labels, title, cv_folds, random_state):
    title = f"==============Treinando com {title}============="
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    for name, clf in classifiers:
        print(name)
        clf.cross_val(data, labels, cv_folds=cv_folds, random_state=random_state)
        print("-------------------------------------\n")


def evaluate_sklearn_models(data_extracted, labels, cfg):
    classifiers = [
        ("SVM", SupportVectorMachine(random_state=cfg.random_state)),
        ("Random Forest", RandomForest(random_state=cfg.random_state)),
        ("XGBoost", XGBoost(n_classes=len(set(labels)), random_state=cfg.random_state)),
    ]
    kw = {"cv_folds": cfg.cv_folds, "random_state": cfg.random_state}

    _train_sklearn(classifiers, data_extracted.data, labels, "toda a base", **kw)

    rgb, rx, hiper = data_extracted.slice_data()
    _train_sklearn(classifiers, rgb, labels, "RGB", **kw)
    _train_sklearn(classifiers, rx, labels, "RX", **kw)
    _train_sklearn(classifiers, hiper, labels, "HIPER", **kw)

    _train_sklearn(classifiers, np.concatenate((rgb, rx), axis=1),
                   labels, "RGB+RX", **kw)
    _train_sklearn(classifiers, np.concatenate((rgb, hiper), axis=1),
                   labels, "RGB+HIPER", **kw)
    _train_sklearn(classifiers, np.concatenate((rx, hiper), axis=1),
                   labels, "RX+HIPER", **kw)


def _run_torch_head(head, data, labels, title, cfg):
    print("\n============================================")
    print(f"==============Treinando com {title}=============")
    print("============================================")
    head.apply(reset_weights)
    do_cross_val_torch(data, labels, head, cfg)


def evaluate_torch_models(data_extracted, labels, head, classifier_index, cfg):
    head = copy.deepcopy(head)
    out_feat = head[classifier_index].out_features

    def _set_in_features(n):
        head[classifier_index] = torch.nn.Linear(
            in_features=n, out_features=out_feat, bias=True
        )

    _set_in_features(data_extracted.data.shape[1])
    _run_torch_head(head, data_extracted.data, labels, "toda a base", cfg)

    rgb, rx, hiper = data_extracted.slice_data()
    _set_in_features(rgb.shape[1]); _run_torch_head(head, rgb, labels, "RGB", cfg)
    _set_in_features(rx.shape[1]);   _run_torch_head(head, rx, labels, "RX", cfg)
    _set_in_features(hiper.shape[1]); _run_torch_head(head, hiper, labels, "HIPER", cfg)

    _set_in_features(np.concatenate((rgb, rx), axis=1).shape[1])
    _run_torch_head(head, np.concatenate((rgb, rx), axis=1),
                    labels, "RGB+RX", cfg)
    _set_in_features(np.concatenate((rgb, hiper), axis=1).shape[1])
    _run_torch_head(head, np.concatenate((rgb, hiper), axis=1),
                    labels, "RGB+HIPER", cfg)
    _set_in_features(np.concatenate((rx, hiper), axis=1).shape[1])
    _run_torch_head(head, np.concatenate((rx, hiper), axis=1),
                    labels, "RX+HIPER", cfg)