import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.classes import DISPLAY_LABELS_EN
from .data import PairedDataset
from .io_utils import load_checkpoint
from .transforms import val_transform
from .training import validate


def plot_confusion_matrix(targets, preds, labels=None):
    labels = labels or DISPLAY_LABELS_EN
    cm = confusion_matrix(targets, preds)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap=plt.cm.Blues)
    plt.show()


def evaluate_folds(dados, rotulos, build_model, streams, checkpoint_pattern, cfg):
    import torch

    kfold = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                            random_state=42)
    score_acc, score_loss = [], []
    targets_all, preds_all = [], []
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)

    for fold, (_, va_idx) in enumerate(
        kfold.split(np.arange(len(dados)), rotulos)
    ):
        net = build_model().to(cfg.device)
        net.load_state_dict(load_checkpoint(
            checkpoint_pattern.format(fold=fold), device=cfg.device))

        val_loader = DataLoader(
            PairedDataset([dados[i] for i in va_idx], rotulos[va_idx],
                          val_transform, streams=streams),
            batch_size=cfg.batch_size, shuffle=False,
        )
        loss, acc, tgt, prd = validate(net, val_loader, criterion, cfg.device)
        score_acc.append(acc)
        score_loss.append(loss)
        targets_all.append(tgt)
        preds_all.append(prd)

        print(f"Fold {fold + 1}: Loss={loss:.4f}, Acc={acc:.4f}")
        del net

    print(f"\n{'-' * 30}")
    print(f"Média Acurácia: {np.mean(score_acc):.4f} "
          f"+/- {np.std(score_acc):.4f}")
    print(f"Média Loss: {np.mean(score_loss):.4f} "
          f"+/- {np.std(score_loss):.4f}")

    targets = np.concatenate(targets_all)
    preds = np.concatenate(preds_all)
    plot_confusion_matrix(targets, preds)
    return targets, preds