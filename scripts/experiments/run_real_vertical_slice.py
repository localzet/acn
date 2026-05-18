from argparse import ArgumentParser, Namespace
from pathlib import Path

from acn.experiments.real_vertical import (
    load_real_vertical_slice_config,
    run_real_vertical_slice,
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run the real ACN adaptive Fashion-MNIST slice.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/acn_real_vertical_slice.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_real_vertical_slice_config(args.config, output_dir=args.output_dir)
    result = run_real_vertical_slice(config)
    print(f"experiment={config.experiment_id}")
    print(f"output_dir={result.artifacts.output_dir}")
    print(f"rollback_restored={result.rollback_restored}")
    print(f"baseline_commit={result.baseline_commit_id}")
    print(f"degraded_commit={result.degraded_commit_id}")
    print(f"recovery_commit={result.recovery_commit_id}")
    print(f"dashboard_snapshot={result.artifacts.dashboard_snapshot_json}")


if __name__ == "__main__":
    main()
