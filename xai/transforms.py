import torchvision.transforms as transforms

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import _IMAGENET_MEAN, _IMAGENET_STD
from common.data import PadToSquare


train_transform = transforms.Compose([
    PadToSquare(fill=0), 
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

val_transform = transforms.Compose([
    PadToSquare(fill=0), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])