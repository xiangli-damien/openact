from __future__ import annotations
from typing import Any, Dict, List, Optional, Type
from openact_eval.evaluators.base import Evaluator
class EvaluatorRegistry:
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
            available = ', '.join(cls.list())
            raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
        return evaluator_cls(**kwargs)
    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._evaluators.keys())
    @classmethod
    def list_with_info(cls) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, evaluator_cls in cls._evaluators.items():
            result[name] = {
                'class': evaluator_cls.__name__,
                'doc': (evaluator_cls.__doc__ or '').strip().split('\n')[0],
            }
        return result
def auto_select_evaluator(
    task_name: str,
    prefer_llm: bool = False,
    llm_backend: str = 'openai',
    llm_model: str = 'gpt-4o-mini',
    guard_model: str = 'meta-llama/Llama-Guard-3-8B',
    guard_device_map: str = 'auto',
    guard_dtype: str = 'auto',
    **kwargs: Any,
) -> Evaluator:
    from openact_eval.eval_descriptor import get_eval_descriptor
    from openact_eval.evaluators.parser_evaluator import ParserEvaluator, TheoremQAEvaluator
    from openact_eval.evaluators.llm_judge import LLMJudgeEvaluator
    from openact_eval.evaluators.safety_evaluators import (
        LlamaGuardEvaluator,
        RefusalHeuristicEvaluator,
    )
    eval_desc = get_eval_descriptor(task_name)
    if eval_desc.eval_strategy == 'safety' and not prefer_llm:
        try:
            import torch
            from transformers import AutoModelForCausalLM
            return LlamaGuardEvaluator(
                model_name=guard_model,
                device_map=guard_device_map,
                dtype=guard_dtype,
                **kwargs,
            )
        except ImportError:
            return RefusalHeuristicEvaluator(**kwargs)
    if prefer_llm or eval_desc.eval_strategy == 'llm_judge':
        return LLMJudgeEvaluator(backend=llm_backend, model=llm_model, **kwargs)
    if eval_desc.eval_strategy == 'parser':
        if task_name.lower() == 'theoremqa':
            return TheoremQAEvaluator()
        return ParserEvaluator(task_name=task_name, **kwargs)
    return LLMJudgeEvaluator(backend=llm_backend, model=llm_model, **kwargs)
