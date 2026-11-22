import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import _IMAGENET_MEAN, _IMAGENET_STD

class PadToSquare:
    """Padding centralizado para deixar a imagem quadrada antes do Resize.

    Mantém o aspecto do conteúdo — o Resize depois só faz upscale uniforme.
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        pad_left   = (side - w) // 2
        pad_right  = side - w - pad_left
        pad_top    = (side - h) // 2
        pad_bottom = side - h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom),
                     fill=self.fill)
    
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