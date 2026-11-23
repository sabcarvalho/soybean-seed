from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class GraphSSL:
    """Wrapper de LabelSpreading com pré-processamento padrão.

    Parâmetros principais:
      kernel        : "knn" (recomendado) ou "rbf"
      n_neighbors   : vizinhos do kernel KNN (só com kernel="knn")
      alpha         : 0.0 = clamping rígido, 1.0 = nenhum clamping
                      (valores típicos: 0.2 a 0.8)
      n_components  : dims do PCA (None = sem PCA)
                      útil para features de alta dim (>500)
      standardize   : padroniza (mean=0, std=1) antes do PCA/kernel
      max_iter      : iterações da propagação
    """

    def __init__(self, kernel="knn", n_neighbors=10, alpha=0.2,
                 n_components=100, standardize=True, max_iter=100,
                 method="spreading"):
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_components = n_components
        self.standardize = standardize
        self.max_iter = max_iter
        self.method = method

        self._model_cls = LabelSpreading if method == "spreading" else LabelPropagation
        self._scaler = None
        self._pca = None
        self._model = None
        self._X_fit = None

    def _preprocess(self, X, fit=False):
        if self.standardize:
            if fit:
                self._scaler = StandardScaler().fit(X)
            X = self._scaler.transform(X)
        if self.n_components is not None and X.shape[1] > self.n_components:
            if fit:
                self._pca = PCA(n_components=self.n_components,
                                random_state=42).fit(X)
            X = self._pca.transform(X)
        return X

    def fit(self, X, y):
        """`y` deve ter -1 para amostras não rotuladas."""
        X = self._preprocess(X, fit=True)
        self._X_fit = X
        self._model = self._model_cls(
            kernel=self.kernel,
            n_neighbors=self.n_neighbors,
            max_iter=self.max_iter,
            n_jobs=1
        ).fit(X, y)
        return self

    def predict(self, X):
        Xp = self._preprocess(X, fit=False)
        if Xp.shape == self._X_fit.shape and np.allclose(Xp, self._X_fit):
            return self._model.transduction_
        return self._model.predict(Xp)

    # ------------------------------------------------------------------ #
    def predict_proba(self, X):
        """Soft labels — útil para pseudo-labels + threshold."""
        Xp = self._preprocess(X, fit=False)
        return self._model.predict_proba(Xp)


# --------------------------------------------------------------------- #
# Cross-validation transdutiva
# --------------------------------------------------------------------- #
def cv_graph_ssl(model_factory, X_labeled, y_labeled, X_unlabeled,
                 cv_folds=5, random_state=42, verbose=True):
    X_all = np.concatenate([X_labeled, X_unlabeled], axis=0) \
        if X_unlabeled is not None and len(X_unlabeled) > 0 else X_labeled
    n_lab = len(X_labeled)

    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)
    scores = []
    for fold, (tr, va) in enumerate(kf.split(X_labeled, y_labeled)):
        # monta y global: -1 nos unlabeled e nos val do fold
        y_masked = np.full(len(X_all), -1, dtype=int)
        train_mask = np.setdiff1d(np.arange(n_lab), va)
        y_masked[train_mask] = y_labeled[train_mask]

        model = model_factory()
        model.fit(X_all, y_masked)
        preds_va = model.predict(X_all)[va]
        acc = accuracy_score(y_labeled[va], preds_va)
        scores.append(acc)
        if verbose:
            print(f"  [GraphSSL fold {fold+1}] acc = {acc:.4f}")

    scores = np.array(scores)
    if verbose:
        print(f"  [GraphSSL] mean = {scores.mean():.4f} "
              f"± {scores.std():.4f}")
    return scores


def evaluate_graph_ssl(data_labeled, y_labeled, data_unlabeled, cfg,
                       model_name, logger=None):
    """Avalia LabelSpreading / LabelPropagation por subconjunto de features.

    Para cada combinação (RGB, RX, HIPER, pares, all), roda CV transdutiva.
    """
    if data_unlabeled is None:
        X_u = np.empty((0, data_labeled.data.shape[1]))
    else:
        X_u = data_unlabeled.data

    splits = _make_feature_splits(data_labeled, X_u)

    configs = {
        "LabelSpreading_knn": dict(kernel="knn", n_neighbors=10, alpha=0.2,
                                   method="spreading"),
        "LabelSpreading_rbf": dict(kernel="rbf", n_neighbors=10, alpha=0.2,
                                   method="spreading"),
        "LabelPropagation_knn": dict(kernel="knn", n_neighbors=10,
                                     method="propagation"),
    }

    for feature_name, (X_l, X_unl) in splits.items():
        for ssl_name, kw in configs.items():
            scores = cv_graph_ssl(
                model_factory=lambda kw=kw: GraphSSL(**kw),
                X_labeled=X_l, y_labeled=y_labeled,
                X_unlabeled=X_unl,
                cv_folds=cfg["classification"]["cv_folds"], random_state=cfg["common"]["seed"],
                verbose=False,
            )
            print(f"  {feature_name:15s} | {ssl_name:25s} | "
                  f"acc={scores.mean():.4f} ± {scores.std():.4f}")
            if logger is not None:
                logger.add(
                    model=model_name, classifier=ssl_name,
                    features=feature_name, mode="graph_ssl",
                    n_labeled=len(X_l), n_unlabeled=len(X_unl),
                    mean_acc=float(scores.mean()), std_acc=float(scores.std()),
                )


def _make_feature_splits(data_labeled, X_unlabeled):
    """Gera (feature_name, (X_labeled_slice, X_unlabeled_slice)) por subconjunto."""
    rgb_l, rx_l, hip_l = data_labeled.slice_data()
    rgb_u, rx_u, hip_u = _slice_external(X_unlabeled, data_labeled)

    splits = {
        "RGB":       (rgb_l, rgb_u),
        "RX":        (rx_l, rx_u),
        "HIPER":     (hip_l, hip_u),
        "RGB+RX":    (np.hstack([rgb_l, rx_l]),   np.hstack([rgb_u, rx_u])),
        "RGB+HIPER": (np.hstack([rgb_l, hip_l]),  np.hstack([rgb_u, hip_u])),
        "RX+HIPER":  (np.hstack([rx_l, hip_l]),   np.hstack([rx_u, hip_u])),
        "all":       (data_labeled.data, X_unlabeled),
    }
    return splits


def _slice_external(X, ref):
    """Fatia X pelas mesmas dimensões de `ref` (RGB|RX|HIPER)."""
    if len(X) == 0:
        return (np.empty((0, ref.data.shape[1] - ref.len_hiper)) ,) * 3
    rgb_d = (X.shape[1] - ref.len_hiper) // 2
    return X[:, :rgb_d], X[:, rgb_d:2*rgb_d], X[:, 2*rgb_d:]