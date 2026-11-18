from dataclasses import dataclass, field
import torch

HYPERSPECTRAL_BANDS = 150
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

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
    random_state: int = 42
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )