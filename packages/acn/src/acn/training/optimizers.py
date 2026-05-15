from collections.abc import Iterable

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from acn.training.config import OptimizerConfig


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    parameters = trainable_parameters(model)
    if not parameters:
        msg = "Optimizer requires at least one trainable parameter."
        raise ValueError(msg)

    return build_optimizer_for_parameters(parameters, config)


def build_optimizer_for_parameters(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> Optimizer:
    parameter_list = list(parameters)
    if not parameter_list:
        msg = "Optimizer requires at least one trainable parameter."
        raise ValueError(msg)

    if config.name == "sgd":
        return SGD(
            parameter_list,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "adam":
        return Adam(
            parameter_list,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.name == "adamw":
        return AdamW(
            parameter_list,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
