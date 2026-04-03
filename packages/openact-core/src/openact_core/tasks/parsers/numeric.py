import re
from typing import Optional
from openact_core.tasks.parsers.base import (
    AnswerParser,
    normalize_numeric,
    extract_boxed,
    find_last_number,
    _RE_NUMBER,
)
from openact_core.tasks.templates import ANSWER_PREFIXES
def _build_answer_prefix_regex() -> str:
    prefixes = sorted(set(ANSWER_PREFIXES.values()) | {"Answer"}, key=len, reverse=True)
    return "|".join(re.escape(p) for p in prefixes)
_ANSWER_PREFIX_RE = _build_answer_prefix_regex()
class NumericParser(AnswerParser):
    _PATTERNS = [
        re.compile(
            rf"(?:^|\n)\s*(?:{_ANSWER_PREFIX_RE})\s*[:=：]\s*(.+?)(?:\n|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"[Tt]he\s+answer\s+is\s*[:=]?\s*(.+?)(?:\n|$)",
        ),
        re.compile(
            r"####\s*(" + _RE_NUMBER + r")",
        ),
    ]
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        prefix_matches = list(self._PATTERNS[0].finditer(response))
        if prefix_matches:
            candidate = prefix_matches[-1].group(1).strip()
            nums = re.findall(_RE_NUMBER, candidate)
            if nums:
                return nums[-1].replace(",", "")
            if candidate:
                return candidate
        answer_is_matches = list(self._PATTERNS[1].finditer(response))
        if answer_is_matches:
            candidate = answer_is_matches[-1].group(1).strip()
            nums = re.findall(_RE_NUMBER, candidate)
            if nums:
                return nums[-1].replace(",", "")
            if candidate:
                return candidate
        hash_matches = list(self._PATTERNS[2].finditer(response))
        if hash_matches:
            return hash_matches[-1].group(1).replace(",", "")
        boxed = extract_boxed(response)
        if boxed is not None:
            num = find_last_number(boxed)
            if num is not None:
                return num
            boxed = boxed.strip()
            if boxed:
                return boxed
        return find_last_number(response)
    def normalize(self, answer: Optional[str]) -> str:
        return normalize_numeric(answer)
