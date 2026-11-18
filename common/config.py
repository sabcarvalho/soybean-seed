from dataclasses import dataclass, field
import torch

@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 20
    lr: float = 1e-3
    weight_decay: float = 8e-3
    cv_folds: int = 5
    early_stopping_patience: int = 4
    unfreeze_after_epoch: int = 5
    unfreeze_lr_scale: float = 0.01
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )