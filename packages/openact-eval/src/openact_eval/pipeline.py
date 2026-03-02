"""
Evaluation pipeline for OpenAct-Eval.

Orchestrates evaluator selection, sample iteration, label saving, and
metrics computation in a single ``run_pipeline()`` call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openact_core.io.run import Run
from openact_eval.evaluators.base import Evaluator, EvalResult
from openact_eval.evaluators.registry import auto_select_evaluator
from openact_eval.metrics import MetricsSummary, compute_metrics


class EvalPipeline:
    """End-to-end evaluation pipeline.

    Parameters
    ----------
    run : Run or str or Path
        The run to evaluate.
    evaluator : Evaluator or None
        Explicit evaluator.  If None, one is auto-selected from the task.
    label_name : str
        Filename stem for the saved label parquet (default ``"correctness"``).
    save_labels : bool
        Whether to persist labels to ``<run>/labels/<label_name>.parquet``.
    only_valid : bool
        If True, only iterate over valid (non-skipped) samples.
    """

    def __init__(
        self,
        run: Union[Run, str, Path],
        evaluator: Optional[Evaluator] = None,
        label_name: str = "correctness",
        save_labels: bool = True,
        only_valid: bool = True,
    ):
        if isinstance(run, Run):
            self._run = run
            self.run_path = Path(run.run_dir)
        else:
            self._run = None
            self.run_path = Path(run)

        self._evaluator = evaluator
        self.label_name = label_name
        self.save_labels = save_labels
        self.only_valid = only_valid
        self._result: Optional[EvalResult] = None
        self._metrics: Optional[MetricsSummary] = None

    @property
    def run(self) -> Run:
        if self._run is None:
            self._run = Run(self.run_path)
        return self._run

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            task_name = self.run.manifest.dataset.name
            if not task_name:
                raise ValueError(
                    "Cannot auto-detect task.  Provide an evaluator explicitly."
                )
            self._evaluator = auto_select_evaluator(task_name)
        return self._evaluator

    def evaluate(
        self,
        sample_indices: Optional[List[int]] = None,
        progress: bool = True,
    ) -> EvalResult:
        """Run evaluation and return the result."""
        self._result = self.evaluator.evaluate(
            run=self.run,
            only_valid=self.only_valid,
            sample_indices=sample_indices,
            progress=progress,
        )
        if self.save_labels:
            path = self._result.save_labels(self.run_path, self.label_name)
            self._result.evaluator_config["label_path"] = path

        self._metrics = compute_metrics(self._result)
        return self._result

    def run_pipeline(
        self,
        sample_indices: Optional[List[int]] = None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Run full pipeline and return a summary dict."""
        result = self.evaluate(sample_indices=sample_indices, progress=progress)
        metrics = self.metrics
        manifest = self.run.manifest

        summary: Dict[str, Any] = {
            "run_path": str(self.run_path),
            "model": manifest.model.name or None,
            "dataset": manifest.dataset.name or None,
            "evaluator": self.evaluator.name,
            "accuracy": metrics.accuracy,
            "n_correct": metrics.n_correct,
            "n_evaluated": metrics.n_evaluated,
            "n_total": metrics.n_total,
            "n_error": metrics.n_error,
            "metrics": metrics.to_dict(),
            "label_path": result.evaluator_config.get("label_path"),
        }

        # attach safety metrics when using a safety evaluator
        if self._is_safety_evaluator():
            from openact_eval.metrics_safety import compute_safety_metrics
            safety = compute_safety_metrics(result)
            summary["safety_metrics"] = safety.to_dict()

        return summary

    @property
    def result(self) -> Optional[EvalResult]:
        return self._result

    @property
    def metrics(self) -> Optional[MetricsSummary]:
        if self._metrics is None and self._result is not None:
            self._metrics = compute_metrics(self._result)
        return self._metrics

    def _is_safety_evaluator(self) -> bool:
        from openact_eval.evaluators.safety_evaluators import SafetyEvaluator
        return isinstance(self.evaluator, SafetyEvaluator)

    def __repr__(self) -> str:
        status = "evaluated" if self._result else "pending"
        return (
            f"EvalPipeline(run='{self.run_path.name}', "
            f"evaluator={self.evaluator.name}, {status})"
        )