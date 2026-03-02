from typing import Any, Dict, Optional

from openact_core.io.sample import Sample
from openact_core.tasks.parsers import AnswerParser, get_parser, list_parsers
from openact_eval.evaluators.base import Evaluator, EvalRecord
from openact_eval.matchers import Matcher, get_matcher


class ParserEvaluator(Evaluator):
    """Deterministic parser-based evaluator.

    Extracts and normalizes answers from model responses, then compares
    against ground truth using a task-appropriate matcher.

    Args:
        task_name: Name of the task (e.g. 'gsm8k', 'mmlu'). Used to look up
            the appropriate parser and matcher via task descriptors.
        parser: Override the auto-selected parser.
        matcher: Override the auto-selected matcher.
        **parser_kwargs: Forwarded to parser construction (e.g. answer_type for theoremqa).
    """

    def __init__(
        self,
        task_name: str,
        parser: Optional[AnswerParser] = None,
        matcher: Optional[Matcher] = None,
        **parser_kwargs: Any,
    ):
        self.task_name = task_name
        self._parser_kwargs = parser_kwargs

        # Try to look up the task descriptor; None if not registered
        descriptor = None
        try:
            from openact_eval.eval_descriptor import get_eval_descriptor
            descriptor = get_eval_descriptor(task_name)
        except (ImportError, ValueError):
            pass

        # If a descriptor exists and says this task needs non-parser evaluation, reject early
        if descriptor and descriptor.eval_strategy != "parser":
            raise ValueError(
                f"Task '{task_name}' requires '{descriptor.eval_strategy}' evaluation strategy. "
                f"Use auto_select_evaluator('{task_name}') instead of ParserEvaluator."
            )

        # Resolve parser and matcher types from descriptor, falling back to task_name / "exact"
        parser_type = (descriptor.parser_type if descriptor else None) or task_name
        matcher_type = (descriptor.matcher_type if descriptor else None) or "exact"

        # Build matcher kwargs (some tasks like theoremqa need answer_type passed through)
        matcher_kwargs = {}
        if task_name.lower() == "theoremqa" and "answer_type" in parser_kwargs:
            matcher_kwargs["answer_type"] = parser_kwargs["answer_type"]

        self._parser = parser or get_parser(parser_type, **parser_kwargs)
        self._matcher = matcher or get_matcher(task_name, matcher_type=matcher_type, **matcher_kwargs)

    @property
    def name(self) -> str:
        return f"parser/{self.task_name}"

    @property
    def parser(self) -> AnswerParser:
        return self._parser

    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        ground_truth = sample.ground_truth
        response_text = sample.response_text

        extracted = self._parser.extract(response_text)
        normalized = self._parser.normalize(extracted) if extracted is not None else None
        normalized_gt = self._parser.normalize(ground_truth) if ground_truth is not None else None

        is_correct = None
        if ground_truth is not None and extracted is not None:
            is_correct = self._matcher.match(normalized, normalized_gt)

        return EvalRecord(
            sample_idx=sample.sample_idx,
            is_correct=is_correct,
            extracted_answer=extracted,
            normalized_answer=normalized,
            ground_truth=ground_truth,
            normalized_ground_truth=normalized_gt,
            score=1.0 if is_correct else (0.0 if is_correct is False else None),
        )

    def _get_config(self) -> Dict[str, Any]:
        return {
            "evaluator_class": self.__class__.__name__,
            "task_name": self.task_name,
            "parser_class": self._parser.__class__.__name__,
            "parser_kwargs": self._parser_kwargs,
        }

    @staticmethod
    def available_tasks() -> list:
        return list_parsers()

    @classmethod
    def from_run(cls, run) -> "ParserEvaluator":
        task_name = run.manifest.dataset.name
        if not task_name:
            raise ValueError(
                "Cannot auto-detect task name from manifest. "
                "Please construct ParserEvaluator with an explicit task_name."
            )
        return cls(task_name=task_name)