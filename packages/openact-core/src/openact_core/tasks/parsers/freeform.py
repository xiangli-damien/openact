import re
from typing import Optional
from openact_core.tasks.parsers.base import AnswerParser, normalize_text
class TruthfulQAParser(AnswerParser):
    _RE_ANSWER_MARKER = re.compile(
        r"[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)"
    )
    _RE_ANSWER_IS = re.compile(
        r"[Tt]he\s+answer\s+is\s*[:=]?\s*(.+?)(?:\.|,|\n|$)",
        re.IGNORECASE,
    )
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        match = self._RE_ANSWER_MARKER.search(response)
        if match:
            return match.group(1).strip()
        match = self._RE_ANSWER_IS.search(response)
        if match:
            return match.group(1).strip()
        lines = response.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line
        return None
    def normalize(self, answer: Optional[str]) -> str:
        return normalize_text(answer)
class IFEvalParser(AnswerParser):
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        return response.strip()
    def normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ""
        return answer.strip()
