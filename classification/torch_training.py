"""Low-level torch training loops + k-fold."""
import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from .data import CustomDataset


def reset_weights(layer):
    if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


def make_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(
        [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}],
        lr=0,
    )


def _epoch_loop(loader, net, criterion, optimizer, device, training):
    net.to(device)
    net.train() if training else net.eval()

    epoch_loss, pred_list, label_list = [], [], []
    ctx = torch.enable_grad if training else torch.no_grad

    with ctx():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            loss = criterion(y_pred, y)
            epoch_loss.append(loss.cpu().data)

            pred = torch.max(y_pred, axis=1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(y.cpu().numpy())

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)
    return accuracy_score(pred, label), float(np.asarray(epoch_loss).mean()), pred, label


def do_cross_val_torch(data, labels, net, cfg):
    """5-fold CV over the (already extracted) feature matrix."""
    device = cfg.device
    criterion = torch.nn.CrossEntropyLoss().to(device)
    dataset = CustomDataset(data=data, labels=labels)
    kfold = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
    )

    score_acc, score_loss, preds, labs = [], [], [], []
    for fold, (tr, va) in enumerate(
        kfold.split(np.arange(len(dataset)), labels)
    ):
        print(f"\rFold {fold + 1}",end="\r", flush=True)
        train_loader = DataLoader(
            Subset(dataset, tr), batch_size=cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, va), batch_size=cfg.batch_size, shuffle=False
        )
        optimizer = make_optimizer(net, cfg.lr, cfg.weight_decay)

        for _ in range(cfg.epochs):
            _epoch_loop(train_loader, net, criterion, optimizer, device, True)

        acc, loss, pred, label = _epoch_loop(val_loader, net, criterion, None, device, False)
        score_acc.append(acc)
        score_loss.append(loss)
        preds.append(pred)
        labs.append(label)
        print(f"\r",end="\r", flush=True)

    print(f"\nAvg accuracy: {np.mean(score_acc)}")
    print(f"Avg loss std: {np.std(score_loss)}")
    return np.array(score_acc), np.array(score_loss), np.array(preds), np.array(labs)