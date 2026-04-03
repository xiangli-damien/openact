import re
from openact_eval.matchers.base import Matcher
from openact_eval.matchers.exact import ExactMatcher
from openact_eval.matchers.numeric import NumericMatcher
from openact_eval.matchers.list_matcher import ListMatcher
class TypeAwareMatcher(Matcher):
    _OPTION_RE = re.compile(r"^\(?([A-Za-z])\)?\.?$")
    def __init__(self, answer_type: str = "float", rel_tol: float = 0.001, abs_tol: float = 1e-06):
        self.answer_type = answer_type.lower().strip()
        self._exact_matcher = ExactMatcher()
        self._numeric_matcher = NumericMatcher(rel_tol=rel_tol, abs_tol=abs_tol)
        self._list_matcher = ListMatcher(rel_tol=rel_tol, abs_tol=abs_tol)
    def match(self, extracted: str, ground_truth: str) -> bool:
        if self.answer_type == "bool":
            return self._match_bool(extracted, ground_truth)
        elif self.answer_type == "option":
            return self._match_option(extracted, ground_truth)
        elif self.answer_type in ("integer", "int", "float"):
            return self._numeric_matcher.match(extracted, ground_truth)
        elif "list" in self.answer_type:
            return self._list_matcher.match(extracted, ground_truth)
        else:
            return self._exact_matcher.match(extracted, ground_truth)
    def _match_bool(self, extracted: str, ground_truth: str) -> bool:
        e = str(extracted).strip().rstrip(".").lower()
        g = str(ground_truth).strip().rstrip(".").lower()
        return e == g
    def _match_option(self, extracted: str, ground_truth: str) -> bool:
        return self._normalize_option(extracted) == self._normalize_option(ground_truth)
    @classmethod
    def _normalize_option(cls, s: str) -> str:
        s = str(s).strip().rstrip(".")
        m = cls._OPTION_RE.match(s)
        if m:
            return m.group(1).upper()
        return s.upper()
