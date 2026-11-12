"""Constants and training configuration."""
from dataclasses import dataclass, field
import torch

CLASS_LABELS = ["Heavy", "Light", "Moderate", "No"]
HYPERSPECTRAL_BANDS = 150


@dataclass
class TorchTrainingConfig:
    epoch_num: int = 10
    lr: float = 1e-3
    weight_decay: float = 8e-3
    batch_size: int = 20
    cv_folds: int = 5
    random_state: int = 42
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )