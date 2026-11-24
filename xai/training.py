"""Fine-tuning e validação."""
import copy, time, os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .data import PairedDataset
from .transforms import train_transform, val_transform
from sklearn.metrics import f1_score

def train_one_epoch(model, dataloader, optimizer, criterion, device, freeze_bn=False):
    model.train()
    if freeze_bn:
        _freeze_bn_stats(model)
    running_loss, correct, total = 0.0, 0, 0

    for k, batch in enumerate(dataloader):
        print("\r--{0}/{1}--".format(k, len(dataloader)), end="", flush=True)
        *inputs, labels = batch
        inputs = [t.to(device) for t in inputs]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            *inputs, labels = batch
            inputs = [t.to(device) for t in inputs]
            labels = labels.to(device)

            outputs = model(*inputs)
            running_loss += criterion(outputs, labels).item() * labels.size(0)

            _, pred = torch.max(outputs, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return running_loss/total, correct/total, macro_f1, all_targets, all_preds

def _unfreeze_and_reoptimize(model, cfg):
    # 1. Congela tudo
    for p in model.encoder_rgb.parameters():
        p.requires_grad = False
    for p in model.encoder_rx.parameters():
        p.requires_grad = False

    # 2. Descongela apenas os N últimos blocos top-level
    n_blocks = getattr(cfg, "unfreeze_last_blocks", 2)
    for encoder in [model.encoder_rgb, model.encoder_rx]:
        blocks = list(encoder.children())
        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    params = [
        {"params": [p for p in model.encoder_rgb.parameters() if p.requires_grad],
         "lr": cfg.lr * cfg.unfreeze_lr_scale},
        {"params": [p for p in model.encoder_rx.parameters() if p.requires_grad],
         "lr": cfg.lr * cfg.unfreeze_lr_scale},
        {"params": model.classifier.parameters(), "lr": cfg.lr},
    ]
    return torch.optim.Adam(params, weight_decay=cfg.weight_decay)

def _freeze_bn_stats(model):
    for encoder in [model.encoder_rgb, model.encoder_rx]:
        for m in encoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval() 

def cross_validate_finetune(dados, rotulos, build_model, streams, cfg,
                             checkpoint_dir, model_tag,
                             train_tf=None, val_tf=None, logger=None):
    
    train_tf = train_tf or train_transform
    val_tf = val_tf or val_transform

    os.makedirs(checkpoint_dir, exist_ok=True)
    kfold = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                            random_state=42)
    score_acc, score_loss = [], []
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)

    for fold, (tr_idx, va_idx) in enumerate(
        kfold.split(np.arange(len(dados)), rotulos)
    ):
        
        print(f"\n{'=' * 30}\nIniciando Fold {fold + 1}\n{'=' * 30}")

        net = build_model().to(cfg.device)
        train_data = [dados[i] for i in tr_idx]
        val_data = [dados[i] for i in va_idx]

        train_loader = DataLoader(
            PairedDataset(train_data, rotulos[tr_idx], train_transform, streams),
            batch_size=cfg.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            PairedDataset(val_data, rotulos[va_idx], val_transform, streams),
            batch_size=cfg.batch_size, shuffle=False,
        )

        for p in net.encoder_rgb.parameters():
            p.requires_grad = False
        for p in net.encoder_rx.parameters():
            p.requires_grad = False

        optim = torch.optim.Adam(net.classifier.parameters(),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_loss, best_state = float("inf"), None
        best_epoch = -1
        patience_counter = 0
        fold_start = time.time()

        for epoch in range(cfg.epochs):
            print(f"\nÉpoca {epoch + 1}/{cfg.epochs}")
            if epoch == cfg.unfreeze_after_epoch:
                print(">>> Descongelando encoders com LR reduzida...")
                optim = _unfreeze_and_reoptimize(net, cfg)

            tr_loss, tr_acc = train_one_epoch(
                net, train_loader, optim, criterion, cfg.device, freeze_bn=(epoch >= cfg.unfreeze_after_epoch))
            va_loss, va_acc, va_f1, _, _= validate(net, val_loader, criterion, cfg.device)
            print(f"\nTrain - Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")
            print(f"Val   - Loss: {va_loss:.4f}, Acc: {va_acc:.4f}")

            if va_loss < best_loss:
                best_loss = va_loss
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Alerta: Paciência {patience_counter}/"
                      f"{cfg.early_stopping_patience}")
                if patience_counter >= cfg.early_stopping_patience:
                    print("Early stopping.")
                    break

        net.load_state_dict(best_state)
        torch.save(net.state_dict(),
                   os.path.join(checkpoint_dir, f"{model_tag}_fold_{fold}.pth"))

        final_loss, final_acc, _, _, _ = validate(net, val_loader, criterion, cfg.device)
        score_acc.append(final_acc)
        score_loss.append(final_loss)
        if logger is not None:
            logger.add(
                model=model_tag,
                fold=fold + 1,
                best_epoch=best_epoch + 1,
                epochs_run=epoch + 1,
                early_stopped=patience_counter >= cfg.early_stopping_patience,
                best_val_loss=float(best_loss),
                final_val_loss=float(final_loss),
                final_val_acc=float(final_acc),
                fold_duration_s=round(time.time() - fold_start, 1),
            )

        del net, optim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'-' * 30}\nResultado da Validação Cruzada:")
    print(f"Média Acurácia: {np.mean(score_acc):.4f} "
          f"+/- {np.std(score_acc):.4f}")
    print(f"Média Loss: {np.mean(score_loss):.4f} "
          f"+/- {np.std(score_loss):.4f}")