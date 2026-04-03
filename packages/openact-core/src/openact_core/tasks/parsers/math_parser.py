import re
from typing import Optional
from openact_core.tasks.parsers.base import (
    AnswerParser,
    normalize_math_expr,
    extract_boxed,
    extract_first_boxed,
    extract_boxed_regex,
    extract_first_boxed_regex,
    find_last_number,
)
_RE_THE_ANSWER_IS = re.compile(
    r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.(?!\d)|\n|$)",
    re.IGNORECASE,
)
_RE_ANSWER_MARKER = re.compile(
    r"[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_CONCLUSION = r"(?:Therefore|Thus|So|Hence)"
_RE_CONCLUSION_IS = re.compile(
    rf"{_CONCLUSION},?\s.*?(?:is|equals?)\s*[=:]?\s*(.+?)(?:\.(?!\d)|\n|$)",
    re.IGNORECASE,
)
_RE_CONCLUSION_EQ = re.compile(
    rf"{_CONCLUSION},?\s[^.]*?=\s*([^\n.]+?)(?:\.(?!\d)|\n|$)",
    re.IGNORECASE,
)
_RE_NOUN_IS = re.compile(
    r"[Tt]he\s+(?:solution|value|sum|area|result|answer)\s+is\s*[=:]?\s*"
    r"(.+?)(?:\.(?!\d)|\n|$)",
    re.IGNORECASE,
)
class MathParser(AnswerParser):
    def __init__(
        self,
        use_first_boxed: bool = False,
        use_regex_boxed: bool = False,
        fallback: bool = True,
    ):
        self.use_first_boxed = use_first_boxed
        self.use_regex_boxed = use_regex_boxed
        self.fallback = fallback
    def _extract_boxed(self, text: str) -> Optional[str]:
        if self.use_regex_boxed:
            fn = extract_first_boxed_regex if self.use_first_boxed else extract_boxed_regex
        else:
            fn = extract_first_boxed if self.use_first_boxed else extract_boxed
        raw = fn(text)
        if raw is None:
            return None
        inner = raw.strip()
        while inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1].strip()
        return inner or None
    @staticmethod
    def _extract_tier1(text: str) -> Optional[str]:
        m = _RE_THE_ANSWER_IS.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
        matches = list(_RE_ANSWER_MARKER.finditer(text))
        if matches and matches[-1].group(1).strip():
            return matches[-1].group(1).strip()
        return None
    @staticmethod
    def _extract_tier2(text: str) -> Optional[str]:
        best: Optional[re.Match] = None
        for pat in (_RE_CONCLUSION_IS, _RE_CONCLUSION_EQ, _RE_NOUN_IS):
            for m in pat.finditer(text):
                if best is None or m.start() > best.start():
                    best = m
        if best is None:
            return None
        candidate = best.group(1).strip()
        if not candidate:
            return None
        num = find_last_number(candidate)
        return num if num is not None else candidate
    @staticmethod
    def _extract_tier3(text: str) -> Optional[str]:
        return find_last_number(text)
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        boxed = self._extract_boxed(response)
        if boxed is not None:
            return boxed
        if not self.fallback:
            return None
        t1 = self._extract_tier1(response)
        if t1 is not None:
            return t1
        t2 = self._extract_tier2(response)
        if t2 is not None:
            return t2
        return self._extract_tier3(response)
    def normalize(self, answer: Optional[str]) -> str:
        return normalize_math_expr(answer)
