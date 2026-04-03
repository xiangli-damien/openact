from __future__ import annotations
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import pandas as pd
from openact_core.io.run import Run
from openact_core.io.sample import Sample

logger = logging.getLogger(__name__)
_CONTEXT_KEYS = (
    'sample_id',
    'language',
    'task_name',
    'task_source',
    'task_split',
    'task_type',
    'prompt_template_name',
    'prompt_template_hash',
    'prompt_template_language',
    'prompt_requested_language',
    'prompt_localization_mode',
    'prompt_answer_prefix',
    'prompt_template_variant',
    'category',
    'level',
    'subject',
    'answer_type',
    'concept',
    'lang_code',
    'behavior_id',
    'split',
    'prompt_variant',
    'attack_method',
    'profile',
    'rep_idx',
    'source',
)


def _is_present(value: Any) -> bool:
    return value not in (None, '', [], {}, ())


def extract_sample_context_meta(sample: Sample) -> Dict[str, Any]:
    raw_meta = sample.meta or {}
    out: Dict[str, Any] = {}
    for key in _CONTEXT_KEYS:
        value = raw_meta.get(key)
        if _is_present(value):
            out[key] = value
    semantic_meta = raw_meta.get('semantic_meta_json') or raw_meta.get('semantic_meta')
    if isinstance(semantic_meta, str) and semantic_meta.strip():
        try:
            semantic_meta = json.loads(semantic_meta)
        except Exception:
            semantic_meta = None
    if isinstance(semantic_meta, dict):
        for key, value in semantic_meta.items():
            if key not in out and _is_present(value):
                out[key] = value
    return out


@dataclass
class EvalRecord:
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
    records: List[EvalRecord]
    evaluator_name: str = ''
    evaluator_config: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_evaluated(self) -> int:
        return sum(record.is_correct is not None for record in self.records)

    @property
    def n_correct(self) -> int:
        return sum(record.is_correct is True for record in self.records)

    @property
    def n_incorrect(self) -> int:
        return sum(record.is_correct is False for record in self.records)

    @property
    def n_error(self) -> int:
        return sum(record.error is not None for record in self.records)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_evaluated if self.n_evaluated else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in self.records:
            row: Dict[str, Any] = {
                'sample_idx': record.sample_idx,
                'is_correct': record.is_correct,
                'extracted_answer': record.extracted_answer,
                'normalized_answer': record.normalized_answer,
                'ground_truth': record.ground_truth,
                'normalized_ground_truth': record.normalized_ground_truth,
                'score': record.score,
                'error': record.error,
            }
            if record.judge_reasoning is not None:
                row['judge_reasoning'] = record.judge_reasoning
            for key, value in record.meta.items():
                row[f'meta_{key}'] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def save_labels(self, run_dir: Union[str, Path], label_name: str = 'correctness') -> str:
        labels_dir = Path(run_dir) / 'labels'
        labels_dir.mkdir(exist_ok=True)
        output_path = labels_dir / f'{label_name}.parquet'
        self.to_dataframe().to_parquet(output_path, index=False)
        return str(output_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'evaluator_name': self.evaluator_name,
            'evaluator_config': self.evaluator_config,
            'accuracy': self.accuracy,
            'n_correct': self.n_correct,
            'n_evaluated': self.n_evaluated,
            'n_total': self.n_total,
            'n_incorrect': self.n_incorrect,
            'n_error': self.n_error,
            'wall_time_s': self.wall_time_s,
            'extra_metrics': self.extra_metrics,
            'records': [asdict(record) for record in self.records],
        }

    def save(self, path: Union[str, Path]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EvalResult':
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        records = [EvalRecord(**record) for record in data.pop('records', [])]
        for transient in ('accuracy', 'n_correct', 'n_evaluated', 'n_total', 'n_incorrect', 'n_error'):
            data.pop(transient, None)
        return cls(records=records, **data)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'evaluator': self.evaluator_name,
            'n_total': self.n_total,
            'n_evaluated': self.n_evaluated,
            'n_correct': self.n_correct,
            'n_incorrect': self.n_incorrect,
            'n_error': self.n_error,
            'accuracy': self.accuracy,
            'wall_time_s': round(self.wall_time_s, 2),
        }
        if self.extra_metrics:
            out.update(self.extra_metrics)
        return out

    def __repr__(self) -> str:
        return f"EvalResult(evaluator={self.evaluator_name!r}, accuracy={self.accuracy:.3f}, n_correct={self.n_correct}/{self.n_evaluated})"


class Evaluator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        ...

    def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    def evaluate(self, run: Run, only_valid: bool = True, sample_indices: Optional[Sequence[int]] = None, progress: bool = True) -> EvalResult:
        self.setup()
        started_at = time.monotonic()
        try:
            iterator = self._build_iterator(run, only_valid, sample_indices)
            if progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, desc=f'Evaluating ({self.name})', total=self._estimate_total(run, only_valid, sample_indices))
                except ImportError:
                    pass
            records: List[EvalRecord] = []
            for sample in iterator:
                try:
                    records.append(self.evaluate_sample(sample))
                except Exception as exc:
                    logger.warning('Error evaluating sample %d: %s', sample.sample_idx, exc)
                    records.append(EvalRecord(sample_idx=sample.sample_idx, error=str(exc), meta=extract_sample_context_meta(sample)))
            return EvalResult(records=records, evaluator_name=self.name, evaluator_config=self._get_config(), wall_time_s=time.monotonic() - started_at, extra_metrics=self._compute_extra_metrics(records))
        finally:
            self.teardown()

    def _build_iterator(self, run: Run, only_valid: bool, sample_indices: Optional[Sequence[int]]) -> Iterator[Sample]:
        if sample_indices is not None:
            for index in sample_indices:
                sample = run[index]
                if only_valid and not sample.is_valid:
                    continue
                yield sample
            return
        if only_valid:
            yield from run.iter_valid()
        else:
            yield from run

    @staticmethod
    def _estimate_total(run: Run, only_valid: bool, sample_indices: Optional[Sequence[int]]) -> Optional[int]:
        if sample_indices is not None:
            return len(sample_indices)
        if only_valid and hasattr(run, 'n_valid'):
            return run.n_valid
        if hasattr(run, '__len__'):
            return len(run)
        return None

    def _compute_extra_metrics(self, records: List[EvalRecord]) -> Dict[str, Any]:
        return {}

    def _get_config(self) -> Dict[str, Any]:
        return {'evaluator_class': self.__class__.__name__}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r})'


class CompositeEvaluator(Evaluator):
    def __init__(self, evaluators: List[Evaluator]):
        if not evaluators:
            raise ValueError('CompositeEvaluator requires at least one sub-evaluator')
        self._evaluators = evaluators

    @property
    def name(self) -> str:
        return 'composite(' + ','.join(evaluator.name for evaluator in self._evaluators) + ')'

    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        raise NotImplementedError('CompositeEvaluator delegates to sub-evaluators via evaluate()')

    def evaluate(self, run: Run, only_valid: bool = True, sample_indices: Optional[Sequence[int]] = None, progress: bool = True) -> EvalResult:
        results = [evaluator.evaluate(run, only_valid=only_valid, sample_indices=sample_indices, progress=progress) for evaluator in self._evaluators]
        return self._merge_results(results)

    def _merge_results(self, results: List[EvalResult]) -> EvalResult:
        if not results:
            return EvalResult(records=[], evaluator_name=self.name)
        primary = results[0]
        extra = {result.evaluator_name: result.summary() for result in results}
        return EvalResult(records=primary.records, evaluator_name=self.name, evaluator_config=self._get_config(), wall_time_s=sum(result.wall_time_s for result in results), extra_metrics=extra)

    def _get_config(self) -> Dict[str, Any]:
        return {'evaluator_class': self.__class__.__name__, 'sub_evaluators': [evaluator._get_config() for evaluator in self._evaluators]}
