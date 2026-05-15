from collections.abc import Iterable

from torch import nn


def freeze_layers(model: nn.Module, layer_prefixes: Iterable[str] | None = None) -> None:
    _set_layers_trainable(model, layer_prefixes, trainable=False)


def unfreeze_layers(model: nn.Module, layer_prefixes: Iterable[str] | None = None) -> None:
    _set_layers_trainable(model, layer_prefixes, trainable=True)


def _set_layers_trainable(
    model: nn.Module,
    layer_prefixes: Iterable[str] | None,
    *,
    trainable: bool,
) -> None:
    prefixes = tuple(layer_prefixes) if layer_prefixes is not None else None
    for name, parameter in model.named_parameters():
        if prefixes is None or name.startswith(prefixes):
            parameter.requires_grad = trainable
