import re
from typing import Any, Dict, Optional, Tuple
from openact_core.io.sample import Sample
from openact_core.tasks.parsers import AnswerParser, get_parser, list_parsers
from openact_eval.evaluators.base import Evaluator, EvalRecord, extract_sample_context_meta
from openact_eval.matchers import Matcher, get_matcher

_ANSWER_TYPE_PATTERNS = [
    re.compile(r'Answer[_ ]type\s*[:=]\s*([^\n\r#]+)', re.IGNORECASE),
    re.compile(r'###\s*Answer[_ ]type\s*[:=]\s*([^\n\r#]+)', re.IGNORECASE),
]


def _infer_answer_type(sample: Sample) -> Tuple[str, str]:
    meta = sample.meta or {}
    for key in ('answer_type', 'Answer_type'):
        value = meta.get(key)
        if value:
            return str(value).strip().lower(), key
    prompt_text = str(meta.get('prompt_text', '') or '')
    for pattern in _ANSWER_TYPE_PATTERNS:
        match = pattern.search(prompt_text)
        if match:
            return match.group(1).strip().lower(), 'prompt_text'
    question = str(meta.get('question', '') or '')
    for pattern in _ANSWER_TYPE_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1).strip().lower(), 'question'
    return 'float', 'default'


def _resolve_ground_truth(sample: Sample) -> Tuple[Optional[str], str]:
    candidates = [('ground_truth', sample.ground_truth)]
    if sample.meta:
        for key in ('answer_key', 'answer', 'answerKey', 'Answer', 'correct_answer', 'correct_answer_num'):
            if key in sample.meta:
                candidates.append((key, sample.meta.get(key)))
    for source, value in candidates:
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            return value_text, source
    return None, 'missing'


class ParserEvaluator(Evaluator):
    def __init__(self, task_name: str, parser: Optional[AnswerParser] = None, matcher: Optional[Matcher] = None, **parser_kwargs: Any):
        self.task_name = task_name
        self._parser_kwargs = parser_kwargs
        descriptor = None
        config = None
        try:
            from openact_eval.eval_descriptor import get_eval_config, get_eval_descriptor
            descriptor = get_eval_descriptor(task_name)
            config = get_eval_config(task_name)
        except (ImportError, ValueError):
            pass
        if descriptor and descriptor.eval_strategy != 'parser':
            raise ValueError(f"Task '{task_name}' requires '{descriptor.eval_strategy}' evaluation strategy. Use auto_select_evaluator('{task_name}') instead of ParserEvaluator.")
        parser_type = (descriptor.parser_type if descriptor else None) or task_name
        matcher_type = (descriptor.matcher_type if descriptor else None) or 'exact'
        matcher_kwargs: Dict[str, Any] = {}
        if config and matcher_type in {'numeric', 'math', 'list', 'type_aware'}:
            matcher_kwargs['rel_tol'] = config.numeric_rel_tol
            matcher_kwargs['abs_tol'] = config.numeric_abs_tol
        if task_name.lower() == 'theoremqa' and 'answer_type' in parser_kwargs:
            matcher_kwargs['answer_type'] = parser_kwargs['answer_type']
        self._parser = parser or get_parser(parser_type, **parser_kwargs)
        self._matcher = matcher or get_matcher(task_name, matcher_type=matcher_type, **matcher_kwargs)

    @property
    def name(self) -> str:
        return f'parser/{self.task_name}'

    @property
    def parser(self) -> AnswerParser:
        return self._parser

    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        if self.task_name.lower() == 'theoremqa' and 'answer_type' not in self._parser_kwargs:
            # A dataset contains mixed answer types. Explicit --evaluator parser
            # must use each row's type just like auto-selection does.
            if not hasattr(self, '_theoremqa_evaluator'):
                self._theoremqa_evaluator = TheoremQAEvaluator()
            return self._theoremqa_evaluator.evaluate_sample(sample)
        ground_truth, ground_truth_source = _resolve_ground_truth(sample)
        response_text = sample.response_text or ''
        extracted = self._parser.extract(response_text)
        parse_failed = extracted is None
        normalized = self._parser.normalize(extracted) if extracted is not None else ''
        normalized_ground_truth = self._parser.normalize(ground_truth) if ground_truth else None
        is_correct = None
        if ground_truth:
            if parse_failed or normalized == '':
                is_correct = False
            else:
                is_correct = self._matcher.match(normalized, normalized_ground_truth)
        meta = extract_sample_context_meta(sample)
        meta.update({'parse_failed': parse_failed, 'gt_missing': ground_truth is None, 'ground_truth_source': ground_truth_source})
        return EvalRecord(sample_idx=sample.sample_idx, is_correct=is_correct, extracted_answer=extracted, normalized_answer=normalized, ground_truth=ground_truth, normalized_ground_truth=normalized_ground_truth, score=1.0 if is_correct else 0.0 if is_correct is False else None, meta=meta)

    def _compute_extra_metrics(self, records):
        parse_failed = sum(1 for record in records if record.meta.get('parse_failed'))
        gt_missing = sum(1 for record in records if record.meta.get('gt_missing'))
        return {'n_parse_failed': parse_failed, 'n_ground_truth_missing': gt_missing}

    def _get_config(self) -> Dict[str, Any]:
        return {'evaluator_class': self.__class__.__name__, 'task_name': self.task_name, 'parser_class': self._parser.__class__.__name__, 'parser_kwargs': self._parser_kwargs}

    @staticmethod
    def available_tasks() -> list:
        return list_parsers()

    @classmethod
    def from_run(cls, run) -> 'ParserEvaluator':
        task_name = run.manifest.dataset.name
        if not task_name:
            raise ValueError('Cannot auto-detect task name from manifest. Please construct ParserEvaluator with an explicit task_name.')
        return cls(task_name=task_name)


class TheoremQAEvaluator(Evaluator):
    def __init__(self):
        self._parser_cache: Dict[str, Any] = {}
        self._matcher_cache: Dict[str, Any] = {}
        self._rel_tol = 0.0
        self._abs_tol = 1e-06
        try:
            from openact_eval.eval_descriptor import get_eval_config
            cfg = get_eval_config('theoremqa')
            self._rel_tol = cfg.numeric_rel_tol
            self._abs_tol = cfg.numeric_abs_tol
        except Exception:
            pass

    @property
    def name(self) -> str:
        return 'parser/theoremqa'

    def _get_parser(self, answer_type: str):
        if answer_type not in self._parser_cache:
            from openact_core.tasks.parsers.theoremqa import TheoremQAParser
            self._parser_cache[answer_type] = TheoremQAParser(answer_type=answer_type)
        return self._parser_cache[answer_type]

    def _get_matcher(self, answer_type: str):
        if answer_type not in self._matcher_cache:
            from openact_eval.matchers.type_aware import TypeAwareMatcher
            self._matcher_cache[answer_type] = TypeAwareMatcher(answer_type=answer_type, rel_tol=self._rel_tol, abs_tol=self._abs_tol)
        return self._matcher_cache[answer_type]

    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        answer_type, answer_type_source = _infer_answer_type(sample)
        parser = self._get_parser(answer_type)
        matcher = self._get_matcher(answer_type)
        ground_truth, ground_truth_source = _resolve_ground_truth(sample)
        response_text = sample.response_text or ''
        extracted = parser.extract(response_text)
        parse_failed = extracted is None
        normalized = parser.normalize(extracted) if extracted is not None else ''
        normalized_ground_truth = parser.normalize(ground_truth) if ground_truth else None
        is_correct = None
        if ground_truth:
            if parse_failed or normalized == '':
                is_correct = False
            else:
                is_correct = matcher.match(normalized, normalized_ground_truth)
        meta = extract_sample_context_meta(sample)
        meta.update({'answer_type': answer_type, 'answer_type_source': answer_type_source, 'parse_failed': parse_failed, 'gt_missing': ground_truth is None, 'ground_truth_source': ground_truth_source})
        return EvalRecord(sample_idx=sample.sample_idx, is_correct=is_correct, extracted_answer=extracted, normalized_answer=normalized, ground_truth=ground_truth, normalized_ground_truth=normalized_ground_truth, score=1.0 if is_correct else 0.0 if is_correct is False else None, meta=meta)

    def _compute_extra_metrics(self, records):
        parse_failed = sum(1 for record in records if record.meta.get('parse_failed'))
        gt_missing = sum(1 for record in records if record.meta.get('gt_missing'))
        return {'n_parse_failed': parse_failed, 'n_ground_truth_missing': gt_missing}

    def _get_config(self) -> Dict[str, Any]:
        return {'evaluator_class': self.__class__.__name__, 'task_name': 'theoremqa', 'rel_tol': self._rel_tol, 'abs_tol': self._abs_tol}
