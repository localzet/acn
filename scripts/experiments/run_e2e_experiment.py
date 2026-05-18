from argparse import ArgumentParser, Namespace
from pathlib import Path

from acn.experiments.e2e import load_e2e_config, run_e2e_experiment


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run a reproducible ACN end-to-end experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/acn_e2e_reproducible.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_e2e_config(args.config, output_dir=args.output_dir)
    result = run_e2e_experiment(config)
    print(f"experiment={config.experiment_id}")
    print(f"output_dir={result.artifacts.output_dir}")
    print(f"stages={len(result.stages)}")
    print(f"rollbacks={len(result.rollback_events)}")
    print(f"branches={len(result.branch_events) + 1}")


if __name__ == "__main__":
    main()
