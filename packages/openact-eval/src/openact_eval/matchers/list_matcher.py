import ast
import re
import math
from typing import Optional, List

from openact_eval.matchers.base import Matcher


class ListMatcher(Matcher):

    def __init__(self, rel_tol: float = 0.001, abs_tol: float = 1e-06):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def match(self, extracted: str, ground_truth: str) -> bool:
        list1 = self._parse_list(extracted)
        list2 = self._parse_list(ground_truth)
        if list1 is None or list2 is None:
            return False
        if len(list1) != len(list2):
            return False
        return all(
            math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            for a, b in zip(list1, list2)
        )

    @staticmethod
    def _parse_list(text: str) -> Optional[List[float]]:
        text = str(text).strip()
        text = text.replace("- ", "-").replace("+ ", "+")
        try:
            result = ast.literal_eval(text)
            if isinstance(result, (list, tuple)):
                return [float(x) for x in result]
        except (ValueError, SyntaxError):
            pass
        cleaned = text.replace(",", " ")
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", cleaned)
        if nums:
            return [float(n) for n in nums]
        return None