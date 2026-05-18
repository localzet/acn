from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch import nn

from acn.artifacts import (
    ArtifactChecksumMismatchError,
    ArtifactNotFoundError,
    LocalArtifactStore,
)
from acn.citadel import CitadelSafetyLayer
from acn.orchestration import RollbackCoordinator
from acn.training import CheckpointManager, CheckpointState
from acn.training.config import OptimizerConfig
from acn.training.optimizers import build_optimizer
from acn.versioning.models import Base
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


def _model() -> nn.Module:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.5)
    return model


def _artifact_path(uri: str) -> Path:
    return Path(unquote(urlparse(uri).path))


def test_rollback_restore_loads_target_checkpoint_state(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    store = LocalArtifactStore(tmp_path)

    with factory() as session:
        repository = SqlAlchemyTrainingVersionRepository(session)
        repository.create_branch(name="main")

        original_model = _model()
        original_optimizer = build_optimizer(original_model, OptimizerConfig())
        saved_weight = original_model.weight.detach().clone()
        checkpoint_manager = CheckpointManager(tmp_path / "checkpoints", artifact_store=store)
        reference = checkpoint_manager.save(
            model=original_model,
            optimizer=original_optimizer,
            scheduler=None,
            scaler=None,
            state=CheckpointState(epoch=3, global_step=30, best_validation_loss=0.2),
            name="main/epoch-0003.pt",
        )
        first_checkpoint = repository.create_checkpoint(
            uri=reference.uri,
            content_hash=reference.checksum,
            size_bytes=reference.size_bytes,
        )
        first_commit = repository.create_commit(
            branch_name="main",
            checkpoint_id=first_checkpoint.id,
            message="stable",
            commit_id="cmt_stable",
        )
        second_checkpoint = repository.create_checkpoint(
            uri="memory://newer",
            content_hash="sha256:newer",
        )
        newer_commit = repository.create_commit(
            branch_name="main",
            checkpoint_id=second_checkpoint.id,
            message="newer",
            commit_id="cmt_newer",
        )

        target_model = _model()
        target_optimizer = build_optimizer(target_model, OptimizerConfig())
        with torch.no_grad():
            target_model.weight.fill_(9.0)
        coordinator = RollbackCoordinator(
            version_repository=repository,
            citadel=CitadelSafetyLayer(version_repository=repository),
            artifact_store=store,
        )

        result = coordinator.rollback_and_restore(
            actor="controller",
            branch_name="main",
            current_commit_id=newer_commit.id,
            target_commit_id=first_commit.id,
            model=target_model,
            optimizer=target_optimizer,
        )

        assert result.branch.head_commit_id == first_commit.id
        assert result.state.epoch == 3
        assert torch.equal(target_model.weight, saved_weight)


def test_rollback_restore_fails_safely_on_missing_artifact(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    store = LocalArtifactStore(tmp_path)

    with factory() as session:
        repository = SqlAlchemyTrainingVersionRepository(session)
        repository.create_branch(name="main")
        checkpoint = repository.create_checkpoint(
            uri=(tmp_path / "missing.pt").as_uri(),
            content_hash="sha256:missing",
        )
        first = repository.create_commit(
            branch_name="main",
            checkpoint_id=checkpoint.id,
            message="first",
            commit_id="cmt_first",
        )
        second_checkpoint = repository.create_checkpoint(
            uri="memory://second",
            content_hash="sha256:second",
        )
        second = repository.create_commit(
            branch_name="main",
            checkpoint_id=second_checkpoint.id,
            message="second",
            commit_id="cmt_second",
        )
        coordinator = RollbackCoordinator(
            version_repository=repository,
            citadel=CitadelSafetyLayer(version_repository=repository),
            artifact_store=store,
        )

        with pytest.raises(ArtifactNotFoundError):
            coordinator.rollback_and_restore(
                actor="controller",
                branch_name="main",
                current_commit_id=second.id,
                target_commit_id=first.id,
                model=_model(),
            )

        assert repository.get_branch("main").head_commit_id == second.id


def test_rollback_restore_fails_safely_on_corrupted_artifact(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    store = LocalArtifactStore(tmp_path)

    with factory() as session:
        repository = SqlAlchemyTrainingVersionRepository(session)
        repository.create_branch(name="main")
        model = _model()
        reference = CheckpointManager(tmp_path / "checkpoints", artifact_store=store).save(
            model=model,
            optimizer=build_optimizer(model, OptimizerConfig()),
            scheduler=None,
            scaler=None,
            state=CheckpointState(epoch=1),
            name="stable.pt",
        )
        checkpoint = repository.create_checkpoint(
            uri=reference.uri,
            content_hash=reference.checksum,
        )
        first = repository.create_commit(
            branch_name="main",
            checkpoint_id=checkpoint.id,
            message="first",
            commit_id="cmt_first",
        )
        second_checkpoint = repository.create_checkpoint(
            uri="memory://second",
            content_hash="sha256:second",
        )
        second = repository.create_commit(
            branch_name="main",
            checkpoint_id=second_checkpoint.id,
            message="second",
            commit_id="cmt_second",
        )
        _artifact_path(reference.uri).write_bytes(b"not a checkpoint")
        coordinator = RollbackCoordinator(
            version_repository=repository,
            citadel=CitadelSafetyLayer(version_repository=repository),
            artifact_store=store,
        )

        with pytest.raises(ArtifactChecksumMismatchError):
            coordinator.rollback_and_restore(
                actor="controller",
                branch_name="main",
                current_commit_id=second.id,
                target_commit_id=first.id,
                model=_model(),
            )

        assert repository.get_branch("main").head_commit_id == second.id
