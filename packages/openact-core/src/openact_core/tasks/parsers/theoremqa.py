"""
Answer parser for TheoremQA.

TheoremQA answers are polymorphic: they can be booleans, option letters,
integers, floats, or lists of numbers.  The ``answer_type`` field from the
dataset drives which normalizer is applied.
"""

import re
import math
import ast
from typing import Optional

from openact_core.tasks.parsers.base import (
    AnswerParser,
    normalize_numeric,
    normalize_letter,
    normalize_text,
)


class TheoremQAParser(AnswerParser):
    """Extract and compare TheoremQA answers with type awareness.

    Parameters
    ----------
    answer_type : str
        One of ``"bool"``, ``"option"``, ``"integer"``, ``"float"``,
        ``"list of integer"``, ``"list of float"``.  Defaults to
        ``"float"``.
    """

    _RE_ANSWER_MARKER = re.compile(
        r"[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)"
    )
    _RE_THEREFORE = re.compile(
        r"[Tt]herefore,?\s*(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|,|\n|$)",
        re.IGNORECASE,
    )

    def __init__(self, answer_type: str = "float"):
        self.answer_type = answer_type.lower().strip()

    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None

        # 1. "Answer: ..."
        match = self._RE_ANSWER_MARKER.search(response)
        if match:
            return match.group(1).strip()

        # 2. "Therefore, the answer is ..."
        match = self._RE_THEREFORE.search(response)
        if match:
            return match.group(1).strip()

        # 3. Fallback: last non-empty line
        lines = response.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line

        return None

    def normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ""
        answer = str(answer).strip()

        if self.answer_type == "bool":
            return self._normalize_bool(answer)
        elif self.answer_type == "option":
            return normalize_letter(answer)
        elif self.answer_type in ("integer", "int"):
            return normalize_numeric(answer)
        elif self.answer_type == "float":
            return self._normalize_float(answer)
        elif "list" in self.answer_type:
            return self._normalize_list(answer)
        else:
            return normalize_text(answer)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_bool(answer: str) -> str:
        low = answer.lower().strip()
        if low in ("true", "yes", "1", "t"):
            return "true"
        if low in ("false", "no", "0", "f"):
            return "false"
        return low

    @staticmethod
    def _normalize_float(answer: str) -> str:
        s = normalize_numeric(answer)
        if not s:
            return ""
        try:
            val = float(s)
            return f"{val:.6f}".rstrip("0").rstrip(".")
        except ValueError:
            return s

    @staticmethod
    def _normalize_list(answer: str) -> str:
        nums = TheoremQAParser._parse_list(answer)
        if nums is None:
            return answer.strip().lower()
        return "[" + ", ".join(str(n) for n in nums) + "]"

    @staticmethod
    def _parse_list(text: str) -> Optional[list]:
        """Try to parse a string as a list of numbers."""
        text = str(text).strip()
        # Try Python literal
        try:
            result = ast.literal_eval(text)
            if isinstance(result, (list, tuple)):
                return [float(x) for x in result]
        except (ValueError, SyntaxError):
            pass
        # Try comma-separated numbers
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        if nums:
            return [float(n) for n in nums]
        return None
