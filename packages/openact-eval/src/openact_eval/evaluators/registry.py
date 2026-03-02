"""
Evaluator registry and automatic evaluator selection.

The registry allows custom evaluators to be registered with a decorator::

    @EvaluatorRegistry.register("my_evaluator")
    class MyEvaluator(Evaluator):
        ...

``auto_select_evaluator`` picks the right evaluator based on the task
descriptor's ``eval_strategy`` field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from openact_eval.evaluators.base import Evaluator


class EvaluatorRegistry:
    """Global evaluator registry."""

    _evaluators: Dict[str, Type[Evaluator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(evaluator_cls: Type[Evaluator]):
            cls._evaluators[name] = evaluator_cls
            return evaluator_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[Evaluator]]:
        return cls._evaluators.get(name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Evaluator:
        evaluator_cls = cls.get(name)
        if evaluator_cls is None:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown evaluator '{name}'.  Available: {available}"
            )
        return evaluator_cls(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._evaluators.keys())

    @classmethod
    def list_with_info(cls) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, evaluator_cls in cls._evaluators.items():
            result[name] = {
                "class": evaluator_cls.__name__,
                "doc": (evaluator_cls.__doc__ or "").strip().split("\n")[0],
            }
        return result


def auto_select_evaluator(
    task_name: str,
    prefer_llm: bool = False,
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o-mini",
    guard_model: str = "meta-llama/Llama-Guard-3-8B",
    guard_device_map: str = "auto",
    guard_dtype: str = "auto",
    **kwargs: Any,
) -> Evaluator:
    """Select the best evaluator for *task_name* automatically.

    Routing logic:
      - ``eval_strategy == "safety"``  -> ``LlamaGuardEvaluator``
      - ``eval_strategy == "parser"``  -> ``ParserEvaluator``
      - ``eval_strategy == "llm_judge"`` or ``prefer_llm`` -> ``LLMJudgeEvaluator``
      - fallback -> ``LLMJudgeEvaluator``
    """
    from openact_eval.eval_descriptor import get_eval_descriptor
    from openact_eval.evaluators.parser_evaluator import ParserEvaluator
    from openact_eval.evaluators.llm_judge import LLMJudgeEvaluator
    from openact_eval.evaluators.safety_evaluators import (
        LlamaGuardEvaluator,
        RefusalHeuristicEvaluator,
    )

    eval_desc = get_eval_descriptor(task_name)

    # safety tasks -> LlamaGuard (primary) or refusal heuristic (fallback)
    if eval_desc.eval_strategy == "safety" and not prefer_llm:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM  # noqa: F401
            return LlamaGuardEvaluator(
                model_name=guard_model,
                device_map=guard_device_map,
                dtype=guard_dtype,
                **kwargs,
            )
        except ImportError:
            # no torch / transformers -> fall back to heuristic
            return RefusalHeuristicEvaluator(**kwargs)

    # explicit LLM preference or descriptor says llm_judge
    if prefer_llm or eval_desc.eval_strategy == "llm_judge":
        return LLMJudgeEvaluator(
            backend=llm_backend, model=llm_model, **kwargs
        )

    # parser-based tasks
    if eval_desc.eval_strategy == "parser":
        return ParserEvaluator(task_name=task_name, **kwargs)

    # default fallback
    return LLMJudgeEvaluator(
        backend=llm_backend, model=llm_model, **kwargs
    )