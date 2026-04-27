from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """CNN backbone used by both baseline and ACN."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        embedding = F.relu(self.fc1(x))
        logits = self.fc2(embedding)

        if return_embedding:
            return logits, embedding
        return logits

    def trainable_layers(self) -> Dict[str, nn.Parameter]:
        """Returns named trainable parameters for layer-wise ACN updates."""
        layers: Dict[str, nn.Parameter] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                layers[name] = param
        return layers
