from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .memory import MemoryBank
from .plasticity import entropy_uncertainty, grad_norms_by_layer, plasticity_scores


class BaseTrainer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))

        return correct / max(total, 1)


class BaselineTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, device: torch.device, lr: float) -> None:
        super().__init__(model, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_stage(
        self,
        loader: torch.utils.data.DataLoader,
        epochs: int,
        desc: str = "Baseline",
        eval_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[List[float], List[float]]:
        epoch_losses: List[float] = []
        epoch_accs: List[float] = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            n_batches = 0

            pbar = tqdm(loader, desc=f"{desc} | epoch {epoch + 1}/{epochs}", leave=False)
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item())
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_losses.append(running_loss / max(n_batches, 1))
            if eval_loader is not None:
                epoch_accs.append(self.evaluate(eval_loader))

        return epoch_losses, epoch_accs


class ACNTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float,
        memory_bank: MemoryBank,
        alpha: float = 1.2,
        beta: float = 0.6,
        gamma: float = 0.8,
    ) -> None:
        super().__init__(model, device)
        self.lr = lr
        self.memory_bank = memory_bank
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.plasticity_history: List[Dict[str, float]] = []

    def train_stage(
        self,
        loader: torch.utils.data.DataLoader,
        epochs: int,
        desc: str = "ACN",
        eval_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        epoch_losses: List[float] = []
        epoch_uncertainty: List[float] = []
        epoch_accs: List[float] = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_uncertainty = 0.0
            n_batches = 0

            pbar = tqdm(loader, desc=f"{desc} | epoch {epoch + 1}/{epochs}", leave=False)
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.model.zero_grad(set_to_none=True)
                logits, embeddings = self.model(images, return_embedding=True)

                loss = self.criterion(logits, targets)
                loss.backward()

                named_params = self.model.trainable_layers()
                uncertainty = float(entropy_uncertainty(logits.detach()).item())
                novelty = float(self.memory_bank.novelty(embeddings.detach()).item())
                grad_norms = grad_norms_by_layer(named_params)
                layer_scores = plasticity_scores(
                    uncertainty=uncertainty,
                    grad_norms=grad_norms,
                    novelty=novelty,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )

                with torch.no_grad():
                    for name, param in named_params.items():
                        if param.grad is None:
                            continue
                        param -= self.lr * layer_scores[name] * param.grad

                self.model.zero_grad(set_to_none=True)

                running_loss += float(loss.item())
                running_uncertainty += uncertainty
                n_batches += 1

                self.plasticity_history.append(layer_scores)
                avg_pl = sum(layer_scores.values()) / max(len(layer_scores), 1)
                pbar.set_postfix(loss=f"{loss.item():.4f}", pl=f"{avg_pl:.3f}")

            epoch_losses.append(running_loss / max(n_batches, 1))
            epoch_uncertainty.append(running_uncertainty / max(n_batches, 1))
            if eval_loader is not None:
                epoch_accs.append(self.evaluate(eval_loader))

        return epoch_losses, epoch_uncertainty, epoch_accs

    def plasticity_series(self) -> Dict[str, List[float]]:
        """Returns per-layer plasticity traces across training steps."""
        traces: Dict[str, List[float]] = defaultdict(list)
        for entry in self.plasticity_history:
            for layer_name, score in entry.items():
                traces[layer_name].append(score)
        return dict(traces)
