
"""Grad-CAM e matrizes de confusão visuais."""
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.classes import DISPLAY_LABELS_PT


class StreamWrapper(torch.nn.Module):
    def __init__(self, base_model, active_idx, n_inputs, fixed_inputs):
        super().__init__()
        self.base_model = base_model
        self.active_idx = active_idx
        self.n_inputs = n_inputs
        for i in range(n_inputs):
            if i != active_idx:
                self.register_buffer(f"fix_{i}", fixed_inputs[i])

    def forward(self, x):
        args = [
            getattr(self, f"fix_{i}") if i != self.active_idx else x
            for i in range(self.n_inputs)
        ]
        return self.base_model(*args)


def coletar_exemplos_matriz(model, dataset, n_classes, device):
    model.eval()
    exemplos = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            *inputs, label = dataset[idx]
            inputs = [t.unsqueeze(0).to(device) for t in inputs]
            outputs = model(*inputs)
            _, pred = torch.max(outputs, dim=1)

            key = (label.item(), pred.item())
            if key not in exemplos:
                exemplos[key] = idx
            if len(exemplos) == n_classes ** 2:
                break
    return exemplos


def _make_wrapper(model, active_idx, inputs, device):
    n_inputs = len(inputs)
    fixed = {
        i: t.unsqueeze(0).to(device)
        for i, t in enumerate(inputs) if i != active_idx
    }
    return StreamWrapper(model, active_idx, n_inputs, fixed)


def _overlay_gradcam(wrapper, target_layer, input_tensor, img_float):
    cam = GradCAM(model=wrapper, target_layers=[target_layer])
    heatmap = cam(input_tensor=input_tensor, targets=None)[0, :]
    return show_cam_on_image(img_float, heatmap, use_rgb=True)


def plotar_matriz_gradcam(model, dataset, exemplos, cfg,
                          stream="rgb", grad=True,
                          stream_keys=("rgb", "rx")):
    """Matriz NxN de exemplos com Grad-CAM sobreposto.

    `stream`       : "rgb" ou "rx" — qual via mostrar.
    `grad`         : se False, mostra a imagem crua (útil como controle).
    `stream_keys`  : ordem dos tensores no dataset — usado para mapear
                     o `active_idx` corretamente.
    """
    n_classes = len(DISPLAY_LABELS_PT)
    active_idx = stream_keys.index(stream)

    fig, axes = plt.subplots(n_classes, n_classes, figsize=(16, 16))
    fig.suptitle(f"Matriz de Confusão Grad-CAM — Via {stream.upper()}",
                 fontsize=24, y=0.98)

    for i in range(n_classes):
        for j in range(n_classes):
            ax = axes[i, j]
            if i == n_classes - 1:
                ax.set_xlabel(f"Predito:\n{DISPLAY_LABELS_PT[j]}",
                              fontsize=14, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Real:\n{DISPLAY_LABELS_PT[i]}",
                              fontsize=14, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

            cor = "limegreen" if i == j else "crimson"

            if (i, j) not in exemplos:
                ax.text(0.5, 0.5, "Sem Ocorrências", ha="center",
                        va="center", fontsize=12, color="gray")
                ax.add_patch(patches.Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    linewidth=4, edgecolor="lightgray", facecolor="whitesmoke",
                ))
                continue

            idx = exemplos[(i, j)]
            caminhos = dataset.dados[idx]
            sample = dataset[idx]
            inputs, _ = sample[:-1], sample[-1]

            wrapper = _make_wrapper(model, active_idx, inputs, cfg.device)
            input_tensor = inputs[active_idx].unsqueeze(0).to(cfg.device)
            target_layer = model.encoder_rgb if stream == "rgb" else model.encoder_rx

            # imagem "base" para o overlay
            if stream == "rgb":
                img_bgr = cv2.imread(caminhos["rgb"])
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_raw = cv2.imread(caminhos["rx"], cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)

            img_float = np.float32(cv2.resize(img, (224, 224))) / 255

            if grad:
                vis = _overlay_gradcam(wrapper, target_layer, input_tensor, img_float)
            else:
                vis = img_float

            ax.imshow(vis)
            ax.add_patch(patches.Rectangle(
                (0, 0), 223, 223, linewidth=8,
                edgecolor=cor, facecolor="none",
            ))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()