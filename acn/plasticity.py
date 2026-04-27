from typing import Dict

import torch


def entropy_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Computes mean predictive entropy for a batch."""
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return entropy.mean()


def grad_norms_by_layer(named_params: Dict[str, torch.nn.Parameter]) -> Dict[str, float]:
    """Returns L2 gradient norms for each layer parameter tensor."""
    norms: Dict[str, float] = {}
    for name, param in named_params.items():
        if param.grad is None:
            norms[name] = 0.0
            continue
        norms[name] = float(torch.norm(param.grad.detach(), p=2).item())
    return norms


def plasticity_scores(
    uncertainty: float,
    grad_norms: Dict[str, float],
    novelty: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, float]:
    """Computes layer-wise plasticity via sigmoid(alpha*u + beta*g + gamma*n)."""

    scores: Dict[str, float] = {}
    for layer_name, grad_norm in grad_norms.items():
        grad_feature = float(torch.log1p(torch.tensor(grad_norm)).item())
        value = alpha * uncertainty + beta * grad_feature + gamma * novelty
        score = float(torch.sigmoid(torch.tensor(value)).item())
        scores[layer_name] = score
    return scores
