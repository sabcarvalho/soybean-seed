from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from xai.data import PairedDataset

def pseudo_label_features(model, X_unlabeled, threshold=0.9,
                          temperature=1.0, balance=True, random_state=42):
    """Prevê X_unlabeled e devolve (mask, pseudo_y).

    mask[i]=True  → amostra i passou do threshold e pode ser usada.
    balance=True  → limita cada classe ao tamanho da classe minoritária
                    já aceita, para não desbalancear o treino.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_unlabeled)
    else:
        with torch.no_grad():
            logits = model(torch.as_tensor(X_unlabeled, dtype=torch.float32))
            proba = F.softmax(logits / temperature, dim=1).cpu().numpy()

    conf = proba.max(axis=1)
    pseudo_y = proba.argmax(axis=1)
    mask = conf >= threshold

    if balance and mask.any():
        rng = np.random.default_rng(random_state)
        keep = np.zeros_like(mask, dtype=bool)
        accepted = np.where(mask)[0]
        classes, counts = np.unique(pseudo_y[accepted], return_counts=True)
        if len(counts) > 0:
            cap = counts.min()
            for c in classes:
                idx_c = accepted[pseudo_y[accepted] == c]
                if len(idx_c) > cap:
                    idx_c = rng.choice(idx_c, size=cap, replace=False)
                keep[idx_c] = True
        mask = keep

    return mask, pseudo_y


def self_train_features(build_model, X_labeled, y_labeled,
                        X_unlabeled, rounds=3, threshold=0.9,
                        confidence_warmup=True, verbose=True):
    """Loop de self-training sobre features.

    Retorna (modelo_final, histórico por rodada).
    """
    X_all = np.array(X_labeled, copy=True)
    y_all = np.array(y_labeled, copy=True)
    history = []

    for r in range(rounds):
        # threshold com warmup: começa mais permissivo e endurece
        t = threshold * (0.75 + 0.25 * r / max(1, rounds - 1)) \
            if confidence_warmup else threshold

        model = build_model()
        model.fit(X_all, y_all)          # treina com o que temos

        if len(X_unlabeled) == 0:
            break

        mask, pseudo_y = pseudo_label_features(
            model, X_unlabeled, threshold=t
        )
        n_new = int(mask.sum())
        history.append({"round": r + 1, "threshold": round(t, 3),
                        "n_pseudo": n_new, "n_total": len(y_all)})

        if verbose:
            print(f"  [SSL round {r+1}] threshold={t:.3f} "
                  f"pseudo={n_new}/{len(X_unlabeled)} "
                  f"total={len(y_all)}")

        if n_new == 0:
            break

        X_all = np.concatenate([X_all, X_unlabeled[mask]], axis=0)
        y_all = np.concatenate([y_all, pseudo_y[mask]], axis=0)

    # treina modelo final com todos os dados aceitos na última rodada
    final = build_model()
    final.fit(X_all, y_all)
    return final, history

class UnlabeledView(Dataset):
    """Envolve uma lista de paths para retornar apenas tensores (sem y).

    Use com `PairedDataset`-style dicts (`{"rgb":..., "rx":..., "hiper":...}`).
    """
    def __init__(self, dados, transform_weak, transform_strong=None):
        self.dados = dados
        self.weak = transform_weak
        self.strong = transform_strong or transform_weak

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        tmp = PairedDataset([self.dados[idx]], [0], self.weak)
        *weak_tensors, _ = tmp[0]

        if self.strong is not self.weak:
            tmp_s = PairedDataset([self.dados[idx]], [0], self.strong)
            *strong_tensors, _ = tmp_s[0]
            return weak_tensors, strong_tensors

        return weak_tensors, weak_tensors


def pseudo_label_batch(model, weak_tensors, threshold=0.95):
    with torch.no_grad():
        logits = model(*weak_tensors)
        proba = F.softmax(logits, dim=1)
    conf, pseudo_y = proba.max(dim=1)
    return conf >= threshold, pseudo_y, proba


def ssl_step_end2end(model, labeled_batch, unlabeled_weak, unlabeled_strong,
                     optimizer, criterion, device,
                     lambda_u=1.0, threshold=0.95):
    model.train()
    optimizer.zero_grad()

    # --- supervised ---
    *x_l, y_l = labeled_batch
    x_l = [t.to(device) for t in x_l]
    y_l = y_l.to(device)
    logits_l = model(*x_l)
    loss_l = criterion(logits_l, y_l)

    # --- unsupervised (pseudo-label via weak view) ---
    weak_dev = [t.to(device) for t in unlabeled_weak]
    strong_dev = [t.to(device) for t in unlabeled_strong]

    with torch.no_grad():
        logits_w = model(*weak_dev)
        proba_w = F.softmax(logits_w, dim=1)
        conf, pseudo_y = proba_w.max(dim=1)
        mask = conf >= threshold

    loss_u = torch.zeros((), device=device)
    if mask.any():
        logits_s = model(*strong_dev)
        loss_u = F.cross_entropy(logits_s[mask], pseudo_y[mask])

    loss = loss_l + lambda_u * loss_u
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_l": float(loss_l.item()),
        "loss_u": float(loss_u.item()),
        "n_pseudo": int(mask.sum()),
        "n_unlabeled": int(len(mask)),
    }


from classification.classifiers import SupportVectorMachine, RandomForest, XGBoost

def _make_feature_splits(data_labeled, y_labeled, data_unlabeled):
    rgb_l, rx_l, hip_l = data_labeled.slice_data()

    if data_unlabeled is None or data_unlabeled.data.shape[0] == 0:
        rgb_u = np.empty((0, rgb_l.shape[1]))
        rx_u  = np.empty((0, rx_l.shape[1]))
        hip_u = np.empty((0, hip_l.shape[1]))
    else:
        rgb_u, rx_u, hip_u = data_unlabeled.slice_data()

    return {
        "all":       (data_labeled.data,          y_labeled,
                      data_unlabeled.data if data_unlabeled is not None
                      else np.empty((0, data_labeled.data.shape[1]))),
        "RGB":       (rgb_l,                      y_labeled, rgb_u),
        "RX":        (rx_l,                       y_labeled, rx_u),
        "HIPER":     (hip_l,                      y_labeled, hip_u),
        "RGB+RX":    (np.hstack([rgb_l, rx_l]),   y_labeled,
                      np.hstack([rgb_u, rx_u])),
        "RGB+HIPER": (np.hstack([rgb_l, hip_l]),  y_labeled,
                      np.hstack([rgb_u, hip_u])),
        "RX+HIPER":  (np.hstack([rx_l, hip_l]),   y_labeled,
                      np.hstack([rx_u, hip_u])),
    }

def evaluate_sklearn_ssl(data_labeled, labels, data_unlabeled, cfg,
                         model_name, logger=None):
    classifiers = {
        "SVM":           lambda: SupportVectorMachine().model,
        "RandomForest":  lambda: RandomForest(random_state=cfg.random_state).model,
        "XGBoost":       lambda: XGBoost(n_classes=len(set(labels)),
                                          random_state=cfg.random_state).model,
    }

    splits = _make_feature_splits(data_labeled, labels, data_unlabeled)

    for feature_name, (X_l, y_l, X_u) in splits.items():
        for clf_name, build in classifiers.items():
            model, hist = self_train_features(
                build, X_l, y_l, X_u,
                rounds=cfg.ssl_rounds,
                threshold=cfg.ssl_threshold,
            )

            from sklearn.model_selection import cross_val_score, StratifiedKFold
            cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                                 random_state=cfg.random_state)
            scores = cross_val_score(model, X_l, y_l, cv=cv, scoring="accuracy")

            if logger is not None:
                logger.add(
                    model=model_name, classifier=clf_name,
                    features=feature_name, mode="ssl",
                    n_labeled=len(y_l), n_unlabeled=len(X_u),
                    mean_acc=float(scores.mean()),
                    std_acc=float(scores.std()),
                    pseudo_rounds=len(hist),
                )