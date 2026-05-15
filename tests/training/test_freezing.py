from torch import nn

from acn.training.freezing import freeze_layers, unfreeze_layers


def test_freeze_and_unfreeze_selected_layers() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    freeze_layers(model, ["0"])

    assert not model[0].weight.requires_grad
    assert not model[0].bias.requires_grad
    assert model[2].weight.requires_grad

    unfreeze_layers(model, ["0"])

    assert model[0].weight.requires_grad
    assert model[0].bias.requires_grad
