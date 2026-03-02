import re
import math
from typing import Optional

from openact_eval.matchers.base import Matcher


class MathMatcher(Matcher):

    def __init__(self, rel_tol: float = 0.001, abs_tol: float = 1e-06):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def match(self, extracted: str, ground_truth: str) -> bool:
        e_norm = self._strip(extracted)
        g_norm = self._strip(ground_truth)
        if e_norm == g_norm:
            return True
        val_ext = self._to_float(extracted)
        val_gt = self._to_float(ground_truth)
        if val_ext is not None and val_gt is not None:
            return math.isclose(val_ext, val_gt, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return False

    @staticmethod
    def _strip(s: str) -> str:
        s = str(s).strip()
        if s.endswith("."):
            s = s[:-1]
        return s

    @staticmethod
    def _to_float(expr: str) -> Optional[float]:
        s = str(expr).strip()
        s = s.lstrip("$").rstrip("%").replace(",", "").strip()
        s = s.replace("- ", "-").replace("+ ", "+")
        if s.endswith("."):
            s = s[:-1]
        s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
        try:
            return float(s)
        except ValueError:
            pass
        if re.fullmatch(r"[0-9+\-*/().eE\s]+", s):
            try:
                node = compile(s, "<expr>", "eval")
                result = eval(node, {"__builtins__": {}}, {})
                return float(result)
            except Exception:
                pass
        return None