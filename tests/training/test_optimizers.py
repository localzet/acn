import pytest
from torch import nn

from acn.training.config import OptimizerConfig, SchedulerConfig
from acn.training.freezing import freeze_layers
from acn.training.optimizers import build_optimizer, trainable_parameters
from acn.training.schedulers import build_scheduler


def test_build_optimizer_uses_only_trainable_parameters() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    freeze_layers(model, ["0"])

    optimizer = build_optimizer(model, OptimizerConfig(name="sgd", learning_rate=0.1))

    assert len(trainable_parameters(model)) == 2
    assert len(optimizer.param_groups[0]["params"]) == 2


def test_build_optimizer_rejects_fully_frozen_model() -> None:
    model = nn.Linear(4, 2)
    freeze_layers(model)

    with pytest.raises(ValueError, match="trainable parameter"):
        build_optimizer(model, OptimizerConfig())


def test_build_scheduler_supports_none() -> None:
    model = nn.Linear(4, 2)
    optimizer = build_optimizer(model, OptimizerConfig())

    scheduler = build_scheduler(optimizer, SchedulerConfig(name="none"))

    assert scheduler is None
