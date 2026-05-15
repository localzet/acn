from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from acn.training import Trainer, TrainerConfig
from acn.training.config import OptimizerConfig, SchedulerConfig
from acn.training.optimizers import build_optimizer
from acn.training.schedulers import build_scheduler


class FashionMnistClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a lightweight ACN Fashion-MNIST baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/fashion-mnist"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    validation_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FashionMnistClassifier()
    optimizer = build_optimizer(
        model,
        OptimizerConfig(name="adamw", learning_rate=args.learning_rate, weight_decay=1e-4),
    )
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="cosine", t_max=args.epochs))
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=TrainerConfig(
            epochs=args.epochs,
            device=args.device,
            mixed_precision=not args.no_mixed_precision,
            max_grad_norm=1.0,
            checkpoint_dir=args.checkpoint_dir,
            log_every_n_steps=100,
        ),
    )
    trainer.fit(train_loader=train_loader, validation_loader=validation_loader)


if __name__ == "__main__":
    main()
