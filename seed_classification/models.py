"""Factory for torchvision models + their feature extractor + head."""
from dataclasses import dataclass

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ModelBundle:
    name: str
    feature_extractor: nn.Module
    preprocess: transforms.Compose
    classifier_head: nn.Module
    classifier_index: int


def _preprocess(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _build_vgg(arch, device, n_multimodal=1174, n_classes=4):
    model = arch(weights="DEFAULT").to(device)
    model.classifier[0] = nn.Linear(in_features=n_multimodal, out_features=4096, bias=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=n_classes, bias=True)
    features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1)).to(device)
    name = "vgg16" if arch is models.vgg16 else "vgg19"
    return ModelBundle(name, features, _preprocess(), model.classifier, 0)


def build_vgg16(device, **kw):
    return _build_vgg(models.vgg16, device, **kw)


def build_vgg19(device, **kw):
    return _build_vgg(models.vgg19, device, **kw)


def build_resnet50(device, n_classes=4):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    features = nn.Sequential(*list(model.children())[:-1]).to(device)
    head = nn.Sequential(nn.Linear(2048, n_classes, bias=True))
    return ModelBundle("resnet50", features, _preprocess(), head, 0)


def build_inception_v3(device, n_classes=4):
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
    model.aux_logits = False
    model.fc = nn.Identity()
    head = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(2048, 1000, bias=True),
    )
    return ModelBundle("inception_v3", model, _preprocess(299), head, 1)


def build_efficientnet_b0(device, n_classes=4):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)
    features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
    return ModelBundle("efficientnet", features, _preprocess(), model.classifier, 1)


def build_mobilenet_v3_large(device, n_classes=4):
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT
    ).to(device)
    features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
    return ModelBundle("mobilenet", features, _preprocess(), model.classifier, 0)


MODEL_BUILDERS = {
    "vgg16": build_vgg16,
    "vgg19": build_vgg19,
    "resnet50": build_resnet50,
    "inception_v3": build_inception_v3,
    "efficientnet_b0": build_efficientnet_b0,
    "mobilenet_v3_large": build_mobilenet_v3_large,
}


def build_model(name, device, **kwargs):
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Modelo desconhecido: {name}")
    return MODEL_BUILDERS[name](device, **kwargs)