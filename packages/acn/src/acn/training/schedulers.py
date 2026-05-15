from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, StepLR

from acn.training.config import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler | None:
    if config.name == "none":
        return None
    if config.name == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    if config.name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.t_max)
    if config.name == "exponential":
        return ExponentialLR(optimizer, gamma=config.gamma)
