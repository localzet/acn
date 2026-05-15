from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from acn.training import CheckpointManager, Trainer, TrainerConfig
from acn.training.config import OptimizerConfig
from acn.training.optimizers import build_optimizer


def _build_loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    inputs = torch.randn(16, 4)
    targets = (inputs.sum(dim=1) > 0).long()
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def _build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_trainer_runs_train_and_validation_loop(tmp_path: Path) -> None:
    model = _build_model()
    optimizer = build_optimizer(model, OptimizerConfig(name="adam", learning_rate=1e-2))
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        config=TrainerConfig(
            epochs=2,
            device="cpu",
            mixed_precision=True,
            checkpoint_dir=tmp_path,
            log_every_n_steps=0,
        ),
    )

    history = trainer.fit(train_loader=_build_loader(), validation_loader=_build_loader())

    assert len(history.train) == 2
    assert len(history.validation) == 2
    assert trainer.state.epoch == 2
    assert trainer.state.global_step == 8
    assert (tmp_path / "epoch-0001.pt").exists()
    assert (tmp_path / "epoch-0002.pt").exists()


def test_checkpoint_round_trip_restores_state(tmp_path: Path) -> None:
    loader = _build_loader()
    model = _build_model()
    optimizer = build_optimizer(model, OptimizerConfig())
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_dir=tmp_path, log_every_n_steps=0),
    )
    trainer.fit(train_loader=loader)

    restored_model = _build_model()
    restored_optimizer = build_optimizer(restored_model, OptimizerConfig())
    restored_trainer = Trainer(
        model=restored_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=restored_optimizer,
        config=TrainerConfig(epochs=1, device="cpu", checkpoint_dir=tmp_path, log_every_n_steps=0),
        checkpoint_manager=CheckpointManager(tmp_path),
    )

    restored_trainer.load_checkpoint(tmp_path / "epoch-0001.pt")

    assert restored_trainer.state.epoch == trainer.state.epoch
    assert restored_trainer.state.global_step == trainer.state.global_step
    for source, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(source, restored)
