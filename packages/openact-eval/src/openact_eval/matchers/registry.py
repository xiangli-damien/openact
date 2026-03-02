from typing import Dict, Type, Optional, Any

from openact_eval.matchers.base import Matcher
from openact_eval.matchers.exact import ExactMatcher
from openact_eval.matchers.numeric import NumericMatcher
from openact_eval.matchers.math import MathMatcher
from openact_eval.matchers.list_matcher import ListMatcher
from openact_eval.matchers.type_aware import TypeAwareMatcher

_MATCHER_TYPE_MAP: Dict[str, Type[Matcher]] = {
    "exact": ExactMatcher,
    "numeric": NumericMatcher,
    "math": MathMatcher,
    "list": ListMatcher,
    "type_aware": TypeAwareMatcher,
}

_TASK_MATCHER_TYPE: Dict[str, str] = {
    "gsm8k": "numeric",
    "mgsm": "numeric",
    "math": "math",
    "mmlu": "exact",
    "belebele": "exact",
    "commonsenseqa": "exact",
    "hotpotqa": "exact",
    "theoremqa": "type_aware",
    "truthfulqa": "exact",
}

_TASK_PARAMS: Dict[str, Dict[str, Any]] = {
    "theoremqa": {"answer_type": "float"},
}


def get_matcher(
    task_name: str,
    matcher_type: Optional[str] = None,
    **kwargs: Any,
) -> Matcher:
    if matcher_type is None:
        try:
            from openact_eval.eval_descriptor import get_eval_descriptor

            eval_desc = get_eval_descriptor(task_name)
            matcher_type = eval_desc.matcher_type
        except (ImportError, ValueError):
            matcher_type = None

    if matcher_type is None:
        matcher_type = _TASK_MATCHER_TYPE.get(task_name.lower(), "exact")

    matcher_cls = _MATCHER_TYPE_MAP.get((matcher_type or "exact").lower())
    if matcher_cls is not None:
        params = _TASK_PARAMS.get(task_name.lower(), {}).copy()
        params.update(kwargs)
        return matcher_cls(**params)
    return ExactMatcher(**kwargs)


MATCHER_REGISTRY = dict(_MATCHER_TYPE_MAP)