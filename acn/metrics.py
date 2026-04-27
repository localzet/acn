from typing import Dict, List


def accuracy_from_counts(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return correct / total


def old_new_accuracy(stage_order: List[str], stage_idx: int, eval_scores: Dict[str, float]) -> Dict[str, float]:
    """Splits accuracy into old-task and new-task metrics."""
    seen = stage_order[: stage_idx + 1]
    new_stage = stage_order[stage_idx]

    old_stages = seen[:-1]
    if old_stages:
        old_acc = sum(eval_scores[s] for s in old_stages) / len(old_stages)
    else:
        old_acc = eval_scores[new_stage]

    return {
        "old_task_accuracy": old_acc,
        "new_task_accuracy": eval_scores[new_stage],
        "mean_seen_accuracy": sum(eval_scores[s] for s in seen) / len(seen),
    }


def forgetting_score(stage_order: List[str], history: List[Dict[str, float]]) -> float:
    """Computes average forgetting over previously seen tasks."""
    if len(history) < 2:
        return 0.0

    worst_drops: List[float] = []
    for stage in stage_order[:-1]:
        scores = [h[stage] for h in history if stage in h]
        if not scores:
            continue
        peak = max(scores)
        current = scores[-1]
        worst_drops.append(max(0.0, peak - current))

    if not worst_drops:
        return 0.0
    return sum(worst_drops) / len(worst_drops)


def adaptation_speed(epoch_scores: List[float], target_ratio: float = 0.9) -> float:
    """Returns first epoch index (1-based) that reaches target_ratio of final accuracy."""
    if not epoch_scores:
        return 0.0
    final_score = max(epoch_scores[-1], 1e-8)
    target = final_score * target_ratio

    for idx, score in enumerate(epoch_scores, start=1):
        if score >= target:
            return float(idx)
    return float(len(epoch_scores))
