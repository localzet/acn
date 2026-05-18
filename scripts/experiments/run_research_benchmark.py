from argparse import ArgumentParser, Namespace
from pathlib import Path

from acn.experiments.research import load_research_benchmark_config, run_research_benchmark


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run ACN research baseline comparisons.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/research_benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_research_benchmark_config(args.config)
    result = run_research_benchmark(config)
    print(f"benchmark={config.benchmark_id}")
    print(f"output_dir={result.artifacts.output_dir}")
    print(f"runs={len(result.runs)}")
    print(f"strategies={len(result.aggregates)}")


if __name__ == "__main__":
    main()
