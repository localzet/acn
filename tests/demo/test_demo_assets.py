import json
from pathlib import Path

import pytest
from scripts.demo.generate_demo_assets import main


def test_demo_asset_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_demo_assets.py",
            "--config",
            "configs/demo/acn_demo_mode.json",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    summary_path = tmp_path / "demo_summary.json"
    screenshot_path = tmp_path / "demo_presentation.svg"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["demo_id"] == "acn-demo-mode"
    assert summary["startup_command"] == "make demo-mode"
    assert screenshot_path.exists()
    assert "<svg" in screenshot_path.read_text(encoding="utf-8")
