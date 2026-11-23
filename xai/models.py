import copy

import torch
import torch.nn as nn
import torchvision.models as models


class TwoStreamVGG19(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.encoder_rgb = nn.Sequential(copy.deepcopy(vgg19.features))
        self.encoder_rx = nn.Sequential(copy.deepcopy(vgg19.features))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088 * 2, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )

    def forward(self, x_rgb, x_rx):
        feat_rgb = torch.flatten(self.encoder_rgb(x_rgb), 1)
        feat_rx = torch.flatten(self.encoder_rx(x_rx), 1)
        return self.classifier(torch.cat((feat_rgb, feat_rx), dim=1))


class ThreeStreamResNet50(nn.Module):
    def __init__(self, num_classes=4, num_hiper_bands=150):
        super().__init__()
        resnet_rgb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet_rx = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.encoder_rgb = nn.Sequential(*list(resnet_rgb.children())[:-2])
        self.encoder_rx = nn.Sequential(*list(resnet_rx.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder_hiper = nn.Sequential(
            nn.Linear(num_hiper_bands, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(in_features=2048 + 2048 + 512,
                                    out_features=num_classes)

    def forward(self, x_rgb, x_rx, x_hiper):
        feat_rgb = torch.flatten(self.pool(self.encoder_rgb(x_rgb)), 1)
        feat_rx = torch.flatten(self.pool(self.encoder_rx(x_rx)), 1)
        feat_hiper = self.encoder_hiper(x_hiper)
        return self.classifier(torch.cat((feat_rgb, feat_rx, feat_hiper), dim=1))

def _inception_encoder():
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.aux_logits = False
    model.AuxLogits = None
    # Seleciona por nome — robusto entre versões do torchvision
    layers = [m for name, m in model.named_children()
              if name not in ("avgpool", "dropout", "fc")]
    return nn.Sequential(*layers)


class ThreeStreamInceptionV3(nn.Module):
    def __init__(self, num_classes=4, num_hiper_bands=150):
        super().__init__()

        self.encoder_rgb = _inception_encoder()
        self.encoder_rx = _inception_encoder()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Encoder espectral — idêntico ao ResNet50
        self.encoder_hiper = nn.Sequential(
            nn.Linear(num_hiper_bands, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # 2048 (RGB) + 2048 (RX) + 512 (Hiper) = 4608
        self.classifier = nn.Linear(in_features=2048 + 2048 + 512,
                                    out_features=num_classes)

    def forward(self, x_rgb, x_rx, x_hiper):
        feat_rgb = torch.flatten(self.pool(self.encoder_rgb(x_rgb)), 1)
        feat_rx = torch.flatten(self.pool(self.encoder_rx(x_rx)), 1)
        feat_hiper = self.encoder_hiper(x_hiper)
        return self.classifier(
            torch.cat((feat_rgb, feat_rx, feat_hiper), dim=1)
        )