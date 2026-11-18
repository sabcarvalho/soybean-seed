import torch


def load_checkpoint(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch <2.4 não conhece `weights_only`
        return torch.load(path, map_location=device)