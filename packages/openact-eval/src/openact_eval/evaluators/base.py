from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from openact_core.io.run import Run
from openact_core.io.sample import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    """Result of evaluating a single sample."""

    sample_idx: int
    is_correct: Optional[bool] = None
    extracted_answer: Optional[str] = None
    normalized_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    normalized_ground_truth: Optional[str] = None
    score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregated result of evaluating a full run (or a subset of samples).

    Core counts are exposed as *computed* properties so they always stay
    consistent with ``records``.  Timing and serialisation helpers that were
    present in the reference design are also included.
    """

    records: List[EvalRecord]
    evaluator_name: str = ""
    evaluator_config: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    # -- Computed properties ------------------------------------------------

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_evaluated(self) -> int:
        return sum(1 for r in self.records if r.is_correct is not None)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.records if r.is_correct is True)

    @property
    def n_incorrect(self) -> int:
        return sum(1 for r in self.records if r.is_correct is False)

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.records if r.error is not None)

    @property
    def accuracy(self) -> float:
        n = self.n_evaluated
        if n == 0:
            return 0.0
        return self.n_correct / n

    # -- DataFrame / Parquet ------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for r in self.records:
            row: Dict[str, Any] = {
                "sample_idx": r.sample_idx,
                "is_correct": r.is_correct,
                "extracted_answer": r.extracted_answer,
                "normalized_answer": r.normalized_answer,
                "ground_truth": r.ground_truth,
                "normalized_ground_truth": r.normalized_ground_truth,
                "score": r.score,
                "error": r.error,
            }
            if r.judge_reasoning is not None:
                row["judge_reasoning"] = r.judge_reasoning
            for k, v in r.meta.items():
                row[f"meta_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def save_labels(self, run_dir: Union[str, Path], label_name: str = "correctness") -> str:
        labels_dir = Path(run_dir) / "labels"
        labels_dir.mkdir(exist_ok=True)
        output_path = labels_dir / f"{label_name}.parquet"
        df = self.to_dataframe()
        df.to_parquet(output_path, index=False)
        return str(output_path)

    # -- JSON serialisation -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "evaluator_config": self.evaluator_config,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_evaluated": self.n_evaluated,
            "n_total": self.n_total,
            "n_incorrect": self.n_incorrect,
            "n_error": self.n_error,
            "wall_time_s": self.wall_time_s,
            "extra_metrics": self.extra_metrics,
            "records": [asdict(r) for r in self.records],
        }

    def save(self, path: Union[str, Path]) -> None:
        """Persist the full result as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvalResult":
        """Load a previously saved JSON result."""
        with open(path) as f:
            data = json.load(f)
        records = [EvalRecord(**r) for r in data.pop("records", [])]
        # Pop computed fields that are not constructor args
        for key in ("accuracy", "n_correct", "n_evaluated", "n_total", "n_incorrect", "n_error"):
            data.pop(key, None)
        return cls(records=records, **data)

    # -- Summary ------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "evaluator": self.evaluator_name,
            "n_total": self.n_total,
            "n_evaluated": self.n_evaluated,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "n_error": self.n_error,
            "accuracy": self.accuracy,
            "wall_time_s": round(self.wall_time_s, 2),
        }
        if self.extra_metrics:
            d.update(self.extra_metrics)
        return d

    def __repr__(self) -> str:
        return (
            f"EvalResult(evaluator='{self.evaluator_name}', "
            f"accuracy={self.accuracy:.3f}, "
            f"n_correct={self.n_correct}/{self.n_evaluated})"
        )


# ---------------------------------------------------------------------------
# Base evaluator
# ---------------------------------------------------------------------------

class Evaluator(ABC):
    """Abstract base for all evaluators.

    Subclasses must implement :pymethod:`name` and :pymethod:`evaluate_sample`.
    Override :pymethod:`setup` / :pymethod:`teardown` for resource management,
    and :pymethod:`_compute_extra_metrics` for task-specific aggregate metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        ...

    # -- Lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        """Called once before evaluation begins. Override to load models etc."""

    def teardown(self) -> None:
        """Called once after evaluation ends (even on error)."""

    # -- Main entry point ---------------------------------------------------

    def evaluate(
        self,
        run: Run,
        only_valid: bool = True,
        sample_indices: Optional[Sequence[int]] = None,
        progress: bool = True,
    ) -> EvalResult:
        self.setup()
        try:
            iterator = self._build_iterator(run, only_valid, sample_indices)

            # Wrap with tqdm if requested
            if progress:
                try:
                    from tqdm import tqdm

                    total = self._estimate_total(run, only_valid, sample_indices)
                    iterator = tqdm(
                        iterator,
                        desc=f"Evaluating ({self.name})",
                        total=total,
                    )
                except ImportError:
                    pass

            records: List[EvalRecord] = []
            t0 = time.monotonic()

            for sample in iterator:
                try:
                    record = self.evaluate_sample(sample)
                except Exception as e:
                    logger.warning(
                        "Error evaluating sample %d: %s", sample.sample_idx, e,
                    )
                    record = EvalRecord(
                        sample_idx=sample.sample_idx,
                        error=str(e),
                    )
                records.append(record)

            wall_time = time.monotonic() - t0

            extra_metrics = self._compute_extra_metrics(records)

            return EvalResult(
                records=records,
                evaluator_name=self.name,
                evaluator_config=self._get_config(),
                wall_time_s=wall_time,
                extra_metrics=extra_metrics,
            )
        finally:
            self.teardown()

    # -- Helpers ------------------------------------------------------------

    def _build_iterator(
        self,
        run: Run,
        only_valid: bool,
        sample_indices: Optional[Sequence[int]],
    ) -> Iterator[Sample]:
        """Yield samples from *run*, respecting ``only_valid`` even when
        explicit ``sample_indices`` are given (fixes a bug in the previous
        implementation which silently included invalid samples when indices
        were specified)."""
        if sample_indices is not None:
            for idx in sample_indices:
                sample = run[idx]
                if only_valid and not sample.is_valid:
                    continue
                yield sample
        elif only_valid:
            yield from run.iter_valid()
        else:
            yield from run

    @staticmethod
    def _estimate_total(
        run: Run,
        only_valid: bool,
        sample_indices: Optional[Sequence[int]],
    ) -> Optional[int]:
        """Best-effort count for the progress bar."""
        if sample_indices is not None:
            return len(sample_indices)
        if only_valid and hasattr(run, "n_valid"):
            return run.n_valid
        if hasattr(run, "__len__"):
            return len(run)
        return None

    def _compute_extra_metrics(
        self, records: List[EvalRecord],
    ) -> Dict[str, Any]:
        """Override to compute task-specific aggregate metrics (e.g. per-
        category accuracy) that will be stored in ``EvalResult.extra_metrics``.
        """
        return {}

    def _get_config(self) -> Dict[str, Any]:
        return {"evaluator_class": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------

class CompositeEvaluator(Evaluator):
    """Run multiple evaluators over the same run and merge results.

    The *primary* evaluator (first in the list) determines the records and
    headline accuracy; metrics from all sub-evaluators are collected into
    ``extra_metrics``.
    """

    def __init__(self, evaluators: List[Evaluator]):
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one sub-evaluator")
        self._evaluators = evaluators

    @property
    def name(self) -> str:
        names = [e.name for e in self._evaluators]
        return "composite(" + ",".join(names) + ")"

    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        raise NotImplementedError(
            "CompositeEvaluator delegates to sub-evaluators via evaluate()"
        )

    def evaluate(
        self,
        run: Run,
        only_valid: bool = True,
        sample_indices: Optional[Sequence[int]] = None,
        progress: bool = True,
    ) -> EvalResult:
        results: List[EvalResult] = []
        for evaluator in self._evaluators:
            result = evaluator.evaluate(
                run,
                only_valid=only_valid,
                sample_indices=sample_indices,
                progress=progress,
            )
            results.append(result)
        return self._merge_results(results)

    def _merge_results(self, results: List[EvalResult]) -> EvalResult:
        if not results:
            return EvalResult(evaluator_name=self.name, records=[])

        primary = results[0]
        merged_extra: Dict[str, Any] = {}
        for r in results:
            merged_extra[r.evaluator_name] = r.summary()

        return EvalResult(
            records=primary.records,
            evaluator_name=self.name,
            evaluator_config=self._get_config(),
            wall_time_s=sum(r.wall_time_s for r in results),
            extra_metrics=merged_extra,
        )

    def _get_config(self) -> Dict[str, Any]:
        return {
            "evaluator_class": self.__class__.__name__,
            "sub_evaluators": [e._get_config() for e in self._evaluators],
        }