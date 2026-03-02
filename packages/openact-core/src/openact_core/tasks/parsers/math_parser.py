import re
from typing import Optional
from openact_core.tasks.parsers.base import AnswerParser, extract_boxed, normalize_math_expr, normalize_numeric


class MathParser(AnswerParser):
    _RE_ANSWER_MARKER = re.compile(
        r'[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)', re.DOTALL
    )
    _RE_ANSWER_IS = re.compile(
        r'[Tt]he\s+answer\s+is\s*[:=]?\s*(.+?)(?:\.(?!\d)|,\s|;\s|\n|$)',
        re.IGNORECASE,
    )
    _RE_EQUALS = re.compile(r'=\s*([^\n=]+?)(?:\n|$)')

    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        match = self._RE_ANSWER_MARKER.search(response)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                inner = extract_boxed(candidate)
                if inner is not None:
                    return inner.strip()
                return candidate
        match = self._RE_ANSWER_IS.search(response)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                inner = extract_boxed(candidate)
                if inner is not None:
                    return inner.strip()
                return candidate
        boxed = extract_boxed(response)
        if boxed is not None:
            return boxed.strip()
        matches = self._RE_EQUALS.findall(response)
        if matches:
            candidate = matches[-1].strip()
            inner = extract_boxed(candidate)
            if inner is not None:
                return inner.strip()
            return candidate
        return None

    def normalize(self, answer: Optional[str]) -> str:
        if answer and '\\boxed{' in answer:
            inner = extract_boxed(answer)
            if inner is not None:
                answer = inner
        return normalize_math_expr(answer)