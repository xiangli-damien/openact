"""
Evaluation metrics and aggregation utilities.

Provides functions to compute standard metrics (accuracy, precision, recall)
from EvalResult objects, as well as grouped/stratified analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from openact_eval.evaluators.base import EvalRecord, EvalResult


@dataclass
class MetricsSummary:
    """Summary statistics from an evaluation."""

    accuracy: float = 0.0
    n_total: int = 0
    n_evaluated: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    n_error: int = 0
    n_no_ground_truth: int = 0
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    group_accuracies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "accuracy": self.accuracy,
            "n_total": self.n_total,
            "n_evaluated": self.n_evaluated,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "n_error": self.n_error,
            "n_no_ground_truth": self.n_no_ground_truth,
        }
        if self.mean_score is not None:
            d["mean_score"] = self.mean_score
            d["std_score"] = self.std_score
        if self.group_accuracies:
            d["group_accuracies"] = self.group_accuracies
        return d

    def __repr__(self) -> str:
        return (
            f"MetricsSummary(accuracy={self.accuracy:.3f}, "
            f"{self.n_correct}/{self.n_evaluated})"
        )


def accuracy(records: Sequence[EvalRecord]) -> float:
    """Compute accuracy from a list of EvalRecords."""
    evaluated = [r for r in records if r.is_correct is not None]
    if not evaluated:
        return 0.0
    return sum(1 for r in evaluated if r.is_correct) / len(evaluated)


def compute_metrics(
    result: EvalResult,
    group_by: Optional[str] = None,
) -> MetricsSummary:
    """
    Compute comprehensive metrics from an EvalResult.

    Args:
        result: The EvalResult from an evaluator.
        group_by: Optional metadata key to group accuracy by
            (e.g. "meta_category", "meta_language").

    Returns:
        MetricsSummary with all computed metrics.
    """
    records = result.records
    n_total = len(records)
    n_error = sum(1 for r in records if r.error is not None)
    n_no_gt = sum(
        1 for r in records
        if r.ground_truth is None and r.is_correct is None and r.error is None
    )
    evaluated = [r for r in records if r.is_correct is not None]
    n_evaluated = len(evaluated)
    n_correct = sum(1 for r in evaluated if r.is_correct)
    n_incorrect = n_evaluated - n_correct
    acc = n_correct / n_evaluated if n_evaluated > 0 else 0.0

    scores = [r.score for r in records if r.score is not None]
    mean_score = float(np.mean(scores)) if scores else None
    std_score = float(np.std(scores)) if scores else None

    group_accs: Dict[str, float] = {}
    if group_by:
        groups: Dict[str, List[EvalRecord]] = {}
        for r in evaluated:
            key = r.meta.get(group_by, r.meta.get(f"meta_{group_by}"))
            if key is None:
                key = "__unknown__"
            key = str(key)
            groups.setdefault(key, []).append(r)

        for g_name, g_records in sorted(groups.items()):
            g_correct = sum(1 for r in g_records if r.is_correct)
            group_accs[g_name] = g_correct / len(g_records) if g_records else 0.0

    return MetricsSummary(
        accuracy=acc,
        n_total=n_total,
        n_evaluated=n_evaluated,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_error=n_error,
        n_no_ground_truth=n_no_gt,
        mean_score=mean_score,
        std_score=std_score,
        group_accuracies=group_accs,
    )


def compare_evaluators(
    results: Dict[str, EvalResult],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple evaluators on the same run.

    Args:
        results: Mapping from evaluator name to EvalResult.

    Returns:
        Comparison table as a dict of dicts.
    """
    comparison = {}
    for name, result in results.items():
        metrics = compute_metrics(result)
        comparison[name] = metrics.to_dict()

    # Agreement analysis between evaluators
    if len(results) >= 2:
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r1 = results[names[i]]
                r2 = results[names[j]]
                idx_map_1 = {r.sample_idx: r for r in r1.records}
                idx_map_2 = {r.sample_idx: r for r in r2.records}
                common = set(idx_map_1.keys()) & set(idx_map_2.keys())
                agree = sum(
                    1 for idx in common
                    if idx_map_1[idx].is_correct == idx_map_2[idx].is_correct
                    and idx_map_1[idx].is_correct is not None
                )
                evaluated_common = sum(
                    1 for idx in common
                    if idx_map_1[idx].is_correct is not None
                    and idx_map_2[idx].is_correct is not None
                )
                agreement = agree / evaluated_common if evaluated_common > 0 else 0.0
                key = f"agreement_{names[i]}_vs_{names[j]}"
                comparison[key] = {
                    "agreement": agreement,
                    "n_common": evaluated_common,
                }

    return comparison
