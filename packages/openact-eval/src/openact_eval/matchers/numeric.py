import math
from openact_eval.matchers.base import Matcher
class NumericMatcher(Matcher):
    def __init__(self, rel_tol: float = 0.001, abs_tol: float = 1e-06):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
    def match(self, extracted: str, ground_truth: str) -> bool:
        e = self._clean(extracted)
        g = self._clean(ground_truth)
        if e == g:
            return True
        try:
            val_ext = float(e)
            val_gt = float(g)
            if self.rel_tol == 0:
                return abs(val_ext - val_gt) <= self.abs_tol
            return math.isclose(val_ext, val_gt, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        except (ValueError, TypeError):
            return False
    @staticmethod
    def _clean(s: str) -> str:
        s = str(s).strip()
        s = s.replace(",", "")
        s = s.replace("- ", "-").replace("+ ", "+")
        if s.endswith("."):
            s = s[:-1]
        return s.strip()
