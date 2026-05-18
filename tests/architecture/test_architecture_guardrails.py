from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACN_ROOT = REPO_ROOT / "packages" / "acn" / "src" / "acn"
WEB_ROOT = REPO_ROOT / "apps" / "web" / "src"


def test_training_layer_does_not_import_api_or_ui() -> None:
    forbidden_prefixes = ("acn_api", "apps.web", "web", "frontend")

    for path in _python_files(ACN_ROOT / "training"):
        imports = _imports(path)
        offenders = [name for name in imports if name.startswith(forbidden_prefixes)]

        assert offenders == [], f"{path} imports forbidden presentation modules: {offenders}"


def test_orchestration_remains_coordinator_only() -> None:
    forbidden_prefixes = ("torch", "torchvision", "acn_api")
    forbidden_names = {"DataLoader", "Dataset", "nn", "Optimizer"}

    for path in _python_files(ACN_ROOT / "orchestration"):
        imports = _imports(path)
        offenders = [
            name
            for name in imports
            if name.startswith(forbidden_prefixes) or name.split(".")[-1] in forbidden_names
        ]

        assert offenders == [], f"{path} imports implementation-specific modules: {offenders}"


def test_dashboard_contracts_remain_typed() -> None:
    contract_path = WEB_ROOT / "types" / "dashboard.ts"
    api_path = WEB_ROOT / "api" / "dashboardApi.ts"
    live_updates_path = WEB_ROOT / "api" / "liveUpdates.ts"

    contract = contract_path.read_text(encoding="utf-8")
    api = api_path.read_text(encoding="utf-8")
    live_updates = live_updates_path.read_text(encoding="utf-8")

    assert "export type DashboardSnapshot" in contract
    assert "export type DashboardEvent" in contract
    assert "Promise<DashboardSnapshot>" in api
    assert "DashboardEvent" in live_updates


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.append(node.module)
            imports.extend(alias.name for alias in node.names)

    return tuple(imports)
