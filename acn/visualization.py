from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_outputs_dir(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)


def plot_baseline_vs_acn_accuracy(
    outputs_dir: Path,
    stage_labels: List[str],
    baseline_series: List[float],
    acn_series: List[float],
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(stage_labels, baseline_series, marker="o", label="Baseline")
    plt.plot(stage_labels, acn_series, marker="o", label="ACN")
    plt.ylim(0.0, 1.0)
    plt.title("Mean Seen-Stage Accuracy")
    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / "baseline_vs_acn_accuracy.png", dpi=140)
    plt.close()


def plot_forgetting(outputs_dir: Path, stage_labels: List[str], baseline_forgetting: List[float], acn_forgetting: List[float]) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(stage_labels, baseline_forgetting, marker="s", label="Baseline")
    plt.plot(stage_labels, acn_forgetting, marker="s", label="ACN")
    plt.title("Forgetting Score Over Stages")
    plt.xlabel("Training Stage")
    plt.ylabel("Forgetting")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / "forgetting_score.png", dpi=140)
    plt.close()


def plot_plasticity(outputs_dir: Path, plasticity_traces: Dict[str, List[float]], max_points: int = 400) -> None:
    plt.figure(figsize=(10, 5))
    for layer_name, scores in plasticity_traces.items():
        if not scores:
            continue
        if len(scores) > max_points:
            idx = np.linspace(0, len(scores) - 1, max_points).astype(int)
            y = np.array(scores)[idx]
            x = np.arange(len(y))
        else:
            y = np.array(scores)
            x = np.arange(len(scores))
        plt.plot(x, y, label=layer_name, alpha=0.85)

    plt.ylim(0.0, 1.0)
    plt.title("ACN Plasticity Over Time")
    plt.xlabel("Training Step (downsampled)")
    plt.ylabel("Plasticity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outputs_dir / "plasticity_over_time.png", dpi=140)
    plt.close()


def plot_stage_examples(outputs_dir: Path, stage_names: List[str], stage_images: List[torch.Tensor]) -> None:
    cols = len(stage_names)
    plt.figure(figsize=(2.2 * cols, 2.4))
    for i, (name, img) in enumerate(zip(stage_names, stage_images), start=1):
        plt.subplot(1, cols, i)
        arr = img.squeeze(0).cpu().numpy()
        plt.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(outputs_dir / "stage_examples.png", dpi=160)
    plt.close()
