from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from classification.classifiers import SupportVectorMachine, RandomForest, XGBoost

sys.path.append(str(Path(__file__).resolve().parent.parent))
from xai.data import PairedDataset


# --------------------------------------------------------------------------- #
# Pseudo-labeling sobre features (numpy / sklearn)
# --------------------------------------------------------------------------- #
def pseudo_label_features(model, X_unlabeled, threshold=0.9,
                          temperature=1.0, balance=True, random_state=42,
                          forward_fn=None):
    """Prevê X_unlabeled e devolve (mask, pseudo_y).

    mask[i]=True  → amostra i passou do threshold e pode ser usada.
    balance=True  → limita cada classe ao tamanho da classe minoritária
                    já aceita, para não desbalancear o treino.

    Parâmetros
    ----------
    forward_fn : callable, opcional
        Necessário apenas quando `model` não expõe `predict_proba` (i.e.,
        é um modelo torch). Deve receber um tensor (n, d) e devolver os
        logits (n, n_classes). Como este módulo trabalha com features
        tabulares (um único array), modelos multimodais devem usar
        `pseudo_label_batch` (mais abaixo).
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_unlabeled), dtype=np.float64)
    elif forward_fn is not None:
        with torch.no_grad():
            logits = forward_fn(torch.as_tensor(X_unlabeled, dtype=torch.float32))
            proba = F.softmax(logits / temperature, dim=1).cpu().numpy()
    else:
        raise TypeError(
            "pseudo_label_features requer um modelo com `predict_proba` "
            "(API sklearn) OU o argumento `forward_fn`. Para modelos torch "
            "multimodais, use `pseudo_label_batch`."
        )

    conf = proba.max(axis=1)
    pseudo_y = proba.argmax(axis=1)
    mask = conf >= threshold

    if balance and mask.any():
        rng = np.random.default_rng(random_state)
        keep = np.zeros_like(mask, dtype=bool)
        accepted = np.where(mask)[0]
        accepted_y = pseudo_y[accepted]                      # cache
        classes, counts = np.unique(accepted_y, return_counts=True)
        if counts.size > 0:
            cap = int(counts.min())
            for c in classes:
                idx_c = accepted[accepted_y == c]
                if len(idx_c) > cap:
                    idx_c = rng.choice(idx_c, size=cap, replace=False)
                keep[idx_c] = True
        mask = keep

    return mask, pseudo_y


# --------------------------------------------------------------------------- #
# Self-training sobre features
# --------------------------------------------------------------------------- #
def self_train_features(build_model, X_labeled, y_labeled,
                        X_unlabeled, rounds=3, threshold=0.9,
                        confidence_warmup=True, verbose=True):
    """Loop de self-training sobre features.

    A cada rodada, o pool não-rotulado é consumido: amostras aceitas são
    retiradas de `X_u`, evitando que o mesmo exemplo seja adicionado
    múltiplas vezes com rótulos potencialmente conflitantes.
    """
    X_all = np.array(X_labeled, copy=True)
    y_all = np.array(y_labeled, copy=True)
    X_u = (np.array(X_unlabeled, copy=True)
           if X_unlabeled is not None else np.empty((0, X_all.shape[1])))
    history = []

    # Nada a fazer: fit simples.
    if X_u.shape[0] == 0:
        final = build_model()
        final.fit(X_all, y_all)
        return final, history

    for r in range(rounds):
        t = (threshold * (0.75 + 0.25 * r / max(1, rounds - 1))
             if confidence_warmup else threshold)

        model = build_model()
        model.fit(X_all, y_all)

        mask, pseudo_y = pseudo_label_features(model, X_u, threshold=t)
        n_new = int(mask.sum())
        history.append({"round": r + 1, "threshold": round(t, 3),
                        "n_pseudo": n_new, "n_total": len(y_all)})

        if verbose:
            print(f"  [SSL round {r+1}] threshold={t:.3f} "
                  f"pseudo={n_new}/{len(X_u)} total={len(y_all)}")

        if n_new == 0:
            break

        X_all = np.concatenate([X_all, X_u[mask]], axis=0)
        y_all = np.concatenate([y_all, pseudo_y[mask]], axis=0)

        # CORREÇÃO: remove aceitas do pool para não duplicar nas próximas
        # rodadas (o modelo pode mudar de opinião → rótulos conflitantes).
        X_u = X_u[~mask]

    final = build_model()
    final.fit(X_all, y_all)
    return final, history


# --------------------------------------------------------------------------- #
# Dataset não-rotulado (torch)
# --------------------------------------------------------------------------- #
class UnlabeledView(Dataset):
    """Envolve uma lista de paths e devolve apenas tensores (sem y).

    Nota: cria um `PairedDataset` de um item por acesso — funciona, mas é
    ineficiente. Para bases grandes, o ideal é adaptar `PairedDataset`
    para aceitar `y=None`.
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


# --------------------------------------------------------------------------- #
# Pseudo-label em batch (torch, multimodal)
# --------------------------------------------------------------------------- #
def pseudo_label_batch(model, weak_tensors, threshold=0.95):
    with torch.no_grad():
        logits = model(*weak_tensors)
        proba = F.softmax(logits, dim=1)
    conf, pseudo_y = proba.max(dim=1)
    return conf >= threshold, pseudo_y, proba


# --------------------------------------------------------------------------- #
# Passo de treino SSL end-to-end (FixMatch)
# --------------------------------------------------------------------------- #
def ssl_step_end2end(model, labeled_batch, unlabeled_weak, unlabeled_strong,
                     optimizer, criterion, device,
                     lambda_u=1.0, threshold=0.95, criterion_u=None):
    """Um passo de FixMatch.

    criterion_u : critério para a perda não-supervisionada.
        Default = F.cross_entropy (sem pesos). Se você usa `criterion`
        com `weight=` para o ramo supervisionado, defina `criterion_u`
        explicitamente se quiser o mesmo peso no ramo não-supervisionado.
    """
    model.train()
    optimizer.zero_grad()

    # --- supervisionado ---
    *x_l, y_l = labeled_batch
    x_l = [t.to(device) for t in x_l]
    y_l = y_l.to(device)
    logits_l = model(*x_l)
    loss_l = criterion(logits_l, y_l)

    # --- não-supervisionado (FixMatch) ---
    weak_dev = [t.to(device) for t in unlabeled_weak]
    strong_dev = [t.to(device) for t in unlabeled_strong]

    mask, pseudo_y, _ = pseudo_label_batch(model, weak_dev, threshold=threshold)

    loss_u = torch.zeros((), device=device)
    if mask.any():
        logits_s = model(*strong_dev)
        crit_u = criterion_u if criterion_u is not None else F.cross_entropy
        loss_u = crit_u(logits_s[mask], pseudo_y[mask])

    loss = loss_l + lambda_u * loss_u
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_l": float(loss_l.item()),
        "loss_u": float(loss_u.item()),
        "n_pseudo": int(mask.sum()),
        "n_unlabeled": int(mask.numel()),
    }


# --------------------------------------------------------------------------- #
# Splits de features
# --------------------------------------------------------------------------- #
def _make_feature_splits(data_labeled, y_labeled, data_unlabeled):
    """Monta visões feature-wise consistentes (concatenção uniforme)."""
    y_labeled = np.asarray(y_labeled)

    rgb_l, rx_l, hip_l = data_labeled.slice_data()
    if data_unlabeled is None or data_unlabeled.data.shape[0] == 0:
        rgb_u = np.empty((0, rgb_l.shape[1]))
        rx_u = np.empty((0, rx_l.shape[1]))
        hip_u = np.empty((0, hip_l.shape[1]))
    else:
        rgb_u, rx_u, hip_u = data_unlabeled.slice_data()

    # "all" agora usa hstack, consistente com as demais combinações.
    # Se `data.data` for diferente de hstack(slice_data()), "all" antes
    # ficava incomparável com "RGB+RX" etc.
    all_l = np.hstack([rgb_l, rx_l, hip_l])
    all_u = np.hstack([rgb_u, rx_u, hip_u])

    return {
        "all":       (all_l,                     y_labeled, all_u),
        "RGB":       (rgb_l,                     y_labeled, rgb_u),
        "RX":        (rx_l,                      y_labeled, rx_u),
        "HIPER":     (hip_l,                     y_labeled, hip_u),
        "RGB+RX":    (np.hstack([rgb_l, rx_l]),  y_labeled,
                      np.hstack([rgb_u, rx_u])),
        "RGB+HIPER": (np.hstack([rgb_l, hip_l]), y_labeled,
                      np.hstack([rgb_u, hip_u])),
        "RX+HIPER":  (np.hstack([rx_l, hip_l]),  y_labeled,
                      np.hstack([rx_u, hip_u])),
    }


# --------------------------------------------------------------------------- #
# Avaliação via CV honesta (SSL dentro de cada fold)
# --------------------------------------------------------------------------- #
def evaluate_sklearn_ssl(data_labeled, labels, data_unlabeled, cfg,
                         model_name, logger=None):
    classifiers = {
        "SVM":          lambda: SupportVectorMachine().model,
        "RandomForest": lambda: RandomForest(random_state=cfg.random_state).model,
        "XGBoost":      lambda: XGBoost(n_classes=len(set(np.asarray(labels))),
                                        random_state=cfg.random_state).model,
    }

    splits = _make_feature_splits(data_labeled, labels, data_unlabeled)

    for feature_name, (X_l, y_l, X_u) in splits.items():
        y_l = np.asarray(y_l)
        _, counts_l = np.unique(y_l, return_counts=True)
        min_per_class = int(counts_l.min()) if counts_l.size else 0
        safe_splits = min(int(cfg.cv_folds), min_per_class)

        for clf_name, build in classifiers.items():
            if safe_splits < 2:
                print(f"  [skip] {feature_name}/{clf_name}: "
                      f"min class count = {min_per_class} < 2")
                if logger is not None:
                    logger.add(
                        model=model_name, classifier=clf_name,
                        features=feature_name, mode="ssl",
                        n_labeled=len(y_l), n_unlabeled=len(X_u),
                        mean_acc=float("nan"), std_acc=float("nan"),
                        pseudo_rounds=0, n_folds=0,
                    )
                continue

            skf = StratifiedKFold(
                n_splits=safe_splits, shuffle=True,
                random_state=cfg.random_state,
            )
            fold_scores, fold_pseudo = [], []

            for tr_idx, val_idx in skf.split(X_l, y_l):
                X_tr, y_tr = X_l[tr_idx], y_l[tr_idx]
                X_val, y_val = X_l[val_idx], y_l[val_idx]

                # CORREÇÃO CRÍTICA: roda SSL no treino do fold,
                # avaliando na validação do fold.
                ssl_model, hist = self_train_features(
                    build, X_tr, y_tr, X_u,
                    rounds=cfg.rounds,
                    threshold=cfg.threshold,
                    verbose=False,
                )

                y_pred = ssl_model.predict(X_val)
                fold_scores.append(float(accuracy_score(y_val, y_pred)))
                fold_pseudo.append(int(sum(h["n_pseudo"] for h in hist)))

            scores = np.asarray(fold_scores)
            if logger is not None:
                logger.add(
                    model=model_name, classifier=clf_name,
                    features=feature_name, mode="ssl",
                    n_labeled=len(y_l), n_unlabeled=len(X_u),
                    mean_acc=float(scores.mean()),
                    std_acc=float(scores.std()),
                    pseudo_rounds=int(np.mean(fold_pseudo)) if fold_pseudo else 0,
                    n_folds=safe_splits,
                )