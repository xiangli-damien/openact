"""
Answer matching strategies for evaluation.

Matchers compare normalized extracted answers against normalized ground truth,
implementing different comparison strategies (exact, numeric tolerance, etc.).
"""

from openact_eval.matchers.base import Matcher
from openact_eval.matchers.exact import ExactMatcher
from openact_eval.matchers.numeric import NumericMatcher
from openact_eval.matchers.math import MathMatcher
from openact_eval.matchers.list_matcher import ListMatcher
from openact_eval.matchers.type_aware import TypeAwareMatcher
from openact_eval.matchers.registry import get_matcher, MATCHER_REGISTRY

__all__ = [
    "Matcher",
    "ExactMatcher",
    "NumericMatcher",
    "MathMatcher",
    "ListMatcher",
    "TypeAwareMatcher",
    "get_matcher",
    "MATCHER_REGISTRY",
]
