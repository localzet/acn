"""Generate deterministic demo playback assets.

Demo mode is a presentation workflow, not a production telemetry source.
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate deterministic ACN demo presentation assets.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo/acn_demo_mode.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "demo_id": config["demo_id"],
        "title": config["title"],
        "seed": config["seed"],
        "steps": config["steps"],
        "startup_command": "make demo-mode",
        "screenshot": "demo_presentation.svg",
    }
    (output_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "demo_presentation.svg").write_text(
        _screenshot_svg(title=str(config["title"]), steps=[str(step) for step in config["steps"]]),
        encoding="utf-8",
    )
    print(f"demo_assets={output_dir}")


def _screenshot_svg(*, title: str, steps: list[str]) -> str:
    rows = "\n".join(
        _text(72, 172 + index * 34, f"{index + 1}. {step}", size=18, fill="#cbd5e1")
        for index, step in enumerate(steps)
    )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">',
            '<rect width="1280" height="720" fill="#070b14"/>',
            '<circle cx="1030" cy="120" r="260" fill="#0ea5e9" opacity="0.18"/>',
            '<circle cx="220" cy="560" r="280" fill="#22c55e" opacity="0.16"/>',
            _text(72, 88, "Adaptive Core Network", size=24, fill="#6ee7b7"),
            _text(72, 134, title, size=42, fill="#f8fafc"),
            rows,
            (
                '<path d="M690 360 C790 210 900 500 1025 280" fill="none" '
                'stroke="#38bdf8" stroke-width="8" stroke-linecap="round" '
                'stroke-dasharray="18 14"/>'
            ),
            '<circle cx="690" cy="360" r="24" fill="#22c55e"/>',
            '<circle cx="830" cy="320" r="24" fill="#38bdf8"/>',
            '<circle cx="1025" cy="280" r="24" fill="#f59e0b"/>',
            _text(690, 430, "commit graph", size=20, fill="#e2e8f0"),
            _text(820, 378, "branch", size=20, fill="#e2e8f0"),
            _text(982, 342, "rollback", size=20, fill="#e2e8f0"),
            "</svg>",
            "",
        ]
    )


def _text(x: int, y: int, value: str, *, size: int, fill: str) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" '
        f'font-family="Arial, sans-serif" font-size="{size}">{value}</text>'
    )


if __name__ == "__main__":
    main()
