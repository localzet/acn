import logging

import pytest

from acn.config.settings import Settings
from acn_worker.main import run


def test_worker_run_configures_logging_and_emits_start_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(env="test", log_level="INFO")

    with caplog.at_level(logging.INFO, logger="acn.worker"):
        run(settings)

    assert any(record.name == "acn.worker" for record in caplog.records)
    assert any(record.message == "worker.started" for record in caplog.records)
