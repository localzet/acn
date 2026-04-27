import argparse
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from acn.memory import MemoryBank
from acn.metrics import adaptation_speed, forgetting_score, old_new_accuracy
from acn.model import SimpleCNN
from acn.trainer import ACNTrainer, BaselineTrainer
from acn.transforms import build_stage_transforms, make_stage_examples
from acn.visualization import (
    ensure_outputs_dir,
    plot_baseline_vs_acn_accuracy,
    plot_forgetting,
    plot_plasticity,
    plot_stage_examples,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def make_stage_loaders(
    root: Path,
    stage_transforms: Dict[str, transforms.Compose],
    batch_size: int,
    seed: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], List[object]]:
    stage_order = ["clean", "rotated", "noisy", "inverted", "blurred"]
    train_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}
    raw_examples: List[object] = []

    generator = torch.Generator().manual_seed(seed)

    for stage in stage_order:
        train_ds = datasets.FashionMNIST(
            root=str(root),
            train=True,
            download=True,
            transform=stage_transforms[stage],
        )
        test_ds = datasets.FashionMNIST(
            root=str(root),
            train=False,
            download=True,
            transform=stage_transforms[stage],
        )

        train_loaders[stage] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=generator,
        )
        test_loaders[stage] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    raw_ds = datasets.FashionMNIST(root=str(root), train=False, download=True)
    raw_examples.append(transforms.ToPILImage()(raw_ds.data[0]))

    return train_loaders, test_loaders, raw_examples


def evaluate_seen_stages(
    trainer,
    stage_order: List[str],
    upto_idx: int,
    test_loaders: Dict[str, DataLoader],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for stage in stage_order[: upto_idx + 1]:
        scores[stage] = trainer.evaluate(test_loaders[stage])
    return scores


def run_baseline(
    model: torch.nn.Module,
    device: torch.device,
    lr: float,
    stage_order: List[str],
    train_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    epochs_clean: int,
    epochs_adapt: int,
):
    trainer = BaselineTrainer(model=model, device=device, lr=lr)

    eval_history: List[Dict[str, float]] = []
    mean_seen: List[float] = []
    old_acc: List[float] = []
    new_acc: List[float] = []
    forgetting: List[float] = []
    adapt_speed: List[float] = []

    for i, stage in enumerate(stage_order):
        epochs = epochs_clean if stage == "clean" else epochs_adapt
        _, epoch_accs = trainer.train_stage(
            train_loaders[stage],
            epochs=epochs,
            desc=f"Baseline-{stage}",
            eval_loader=test_loaders[stage],
        )

        stage_scores = evaluate_seen_stages(trainer, stage_order, i, test_loaders)
        eval_history.append(stage_scores)

        split = old_new_accuracy(stage_order, i, stage_scores)
        mean_seen.append(split["mean_seen_accuracy"])
        old_acc.append(split["old_task_accuracy"])
        new_acc.append(split["new_task_accuracy"])
        forgetting.append(forgetting_score(stage_order[: i + 1], eval_history))
        adapt_speed.append(adaptation_speed(epoch_accs) if epoch_accs else float(epochs))

    return {
        "trainer": trainer,
        "eval_history": eval_history,
        "mean_seen": mean_seen,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "forgetting": forgetting,
        "adapt_speed": adapt_speed,
    }


def run_acn(
    model: torch.nn.Module,
    device: torch.device,
    lr: float,
    stage_order: List[str],
    train_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    epochs_clean: int,
    epochs_adapt: int,
):
    memory_bank = MemoryBank(embedding_dim=128, device=device)
    trainer = ACNTrainer(model=model, device=device, lr=lr, memory_bank=memory_bank)

    eval_history: List[Dict[str, float]] = []
    mean_seen: List[float] = []
    old_acc: List[float] = []
    new_acc: List[float] = []
    forgetting: List[float] = []
    adapt_speed: List[float] = []

    for i, stage in enumerate(stage_order):
        epochs = epochs_clean if stage == "clean" else epochs_adapt
        _, _, epoch_accs = trainer.train_stage(
            train_loaders[stage],
            epochs=epochs,
            desc=f"ACN-{stage}",
            eval_loader=test_loaders[stage],
        )

        trainer.memory_bank.update_from_loader(trainer.model, train_loaders[stage])

        stage_scores = evaluate_seen_stages(trainer, stage_order, i, test_loaders)
        eval_history.append(stage_scores)

        split = old_new_accuracy(stage_order, i, stage_scores)
        mean_seen.append(split["mean_seen_accuracy"])
        old_acc.append(split["old_task_accuracy"])
        new_acc.append(split["new_task_accuracy"])
        forgetting.append(forgetting_score(stage_order[: i + 1], eval_history))
        adapt_speed.append(adaptation_speed(epoch_accs) if epoch_accs else float(epochs))

    return {
        "trainer": trainer,
        "eval_history": eval_history,
        "mean_seen": mean_seen,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "forgetting": forgetting,
        "adapt_speed": adapt_speed,
        "plasticity_traces": trainer.plasticity_series(),
    }


def print_summary_table(stage_order: List[str], baseline_res: Dict, acn_res: Dict) -> None:
    print("\n" + "=" * 96)
    print("SUMMARY TABLE")
    print("=" * 96)
    header = (
        f"{'Stage':<10}"
        f"{'B_old':>10}{'B_new':>10}{'B_forget':>12}{'B_speed':>10}"
        f"{'A_old':>10}{'A_new':>10}{'A_forget':>12}{'A_speed':>10}"
    )
    print(header)
    print("-" * 96)

    for i, stage in enumerate(stage_order):
        row = (
            f"{stage:<10}"
            f"{baseline_res['old_acc'][i]:>10.4f}"
            f"{baseline_res['new_acc'][i]:>10.4f}"
            f"{baseline_res['forgetting'][i]:>12.4f}"
            f"{baseline_res['adapt_speed'][i]:>10.2f}"
            f"{acn_res['old_acc'][i]:>10.4f}"
            f"{acn_res['new_acc'][i]:>10.4f}"
            f"{acn_res['forgetting'][i]:>12.4f}"
            f"{acn_res['adapt_speed'][i]:>10.2f}"
        )
        print(row)

    print("-" * 96)
    print(
        f"{'Final mean seen acc':<24}"
        f"Baseline={baseline_res['mean_seen'][-1]:.4f} | "
        f"ACN={acn_res['mean_seen'][-1]:.4f}"
    )
    print(
        f"{'Final forgetting':<24}"
        f"Baseline={baseline_res['forgetting'][-1]:.4f} | "
        f"ACN={acn_res['forgetting'][-1]:.4f}"
    )
    print("=" * 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Core Network (ACN) prototype")
    parser.add_argument("--epochs-clean", type=int, default=3)
    parser.add_argument("--epochs-adapt", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / "outputs"
    data_dir = project_root / "data"
    ensure_outputs_dir(outputs_dir)

    stage_order = ["clean", "rotated", "noisy", "inverted", "blurred"]
    stage_transforms = build_stage_transforms(seed=args.seed)
    train_loaders, test_loaders, raw_examples = make_stage_loaders(
        root=data_dir,
        stage_transforms=stage_transforms,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    stage_names, stage_images = make_stage_examples(raw_examples, stage_transforms)
    plot_stage_examples(outputs_dir, stage_names, stage_images)

    baseline_model = SimpleCNN().to(device)
    acn_model = SimpleCNN().to(device)
    acn_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))

    baseline_res = run_baseline(
        model=baseline_model,
        device=device,
        lr=args.lr,
        stage_order=stage_order,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_clean=args.epochs_clean,
        epochs_adapt=args.epochs_adapt,
    )

    acn_res = run_acn(
        model=acn_model,
        device=device,
        lr=args.lr,
        stage_order=stage_order,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_clean=args.epochs_clean,
        epochs_adapt=args.epochs_adapt,
    )

    plot_baseline_vs_acn_accuracy(
        outputs_dir,
        stage_labels=stage_order,
        baseline_series=baseline_res["mean_seen"],
        acn_series=acn_res["mean_seen"],
    )
    plot_forgetting(
        outputs_dir,
        stage_labels=stage_order,
        baseline_forgetting=baseline_res["forgetting"],
        acn_forgetting=acn_res["forgetting"],
    )
    plot_plasticity(outputs_dir, acn_res["plasticity_traces"])

    print_summary_table(stage_order, baseline_res, acn_res)
    print(f"\nSaved plots to: {outputs_dir}")


if __name__ == "__main__":
    main()
