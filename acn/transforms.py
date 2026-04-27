from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class AddGaussianNoise:
    """Adds Gaussian noise to tensor images."""

    def __init__(self, std: float = 0.2) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class InvertTensor:
    """Inverts grayscale tensor values in [0, 1]."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return 1.0 - tensor


def build_stage_transforms(seed: int) -> Dict[str, transforms.Compose]:
    """Builds deterministic transform pipelines for stream stages."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    base = [transforms.ToTensor()]
    return {
        "clean": transforms.Compose(base),
        "rotated": transforms.Compose(
            [
                transforms.RandomRotation(degrees=35, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        ),
        "noisy": transforms.Compose([transforms.ToTensor(), AddGaussianNoise(std=0.25)]),
        "inverted": transforms.Compose([transforms.ToTensor(), InvertTensor()]),
        "blurred": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.8, 1.4)),
            ]
        ),
    }


def make_stage_examples(images: List[Image.Image], stage_transforms: Dict[str, transforms.Compose]) -> Tuple[List[str], List[torch.Tensor]]:
    """Applies each stage transform to one reference image for visualization."""

    stages = ["clean", "rotated", "noisy", "inverted", "blurred"]
    sample_img = images[0]
    outputs: List[torch.Tensor] = []
    for stage in stages:
        tensor_img = stage_transforms[stage](sample_img)
        outputs.append(tensor_img)
    return stages, outputs
