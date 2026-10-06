"""Shared deterministic CNN; the last layer returns raw logits."""
from copy import deepcopy
import torch
from torch import nn
from config import METHODS


class ReducedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(288, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )
        self.to(device="cpu", dtype=torch.float32)

    def forward(self, x):
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("Protocol requires float32 CPU inputs")
        return self.classifier(self.features(x))


def paired_models(seed):
    torch.manual_seed(seed)
    template = ReducedCNN()
    return {method: deepcopy(template) for method in METHODS}
