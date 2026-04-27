from typing import Optional

import torch


class MemoryBank:
    """Stores a global embedding centroid and exposes novelty scores."""

    def __init__(self, embedding_dim: int, device: torch.device) -> None:
        self.embedding_dim = embedding_dim
        self.device = device
        self.centroid: Optional[torch.Tensor] = None

    def novelty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns batch novelty as mean distance to centroid."""
        if self.centroid is None:
            return torch.tensor(0.0, device=embeddings.device)

        center = self.centroid.to(embeddings.device)
        distances = torch.norm(embeddings - center, p=2, dim=1)
        score = distances.mean()
        norm = torch.norm(center, p=2) + 1e-6
        return score / norm

    @torch.no_grad()
    def update_from_loader(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> None:
        """Updates centroid from all embeddings in a stage loader."""
        model.eval()
        all_embeddings = []

        for images, _ in loader:
            images = images.to(self.device)
            _, emb = model(images, return_embedding=True)
            all_embeddings.append(emb.detach().cpu())

        if not all_embeddings:
            return

        stacked = torch.cat(all_embeddings, dim=0)
        self.centroid = stacked.mean(dim=0).to(self.device)
