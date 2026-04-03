import re
from typing import Optional, Set
from openact_core.tasks.parsers.base import AnswerParser, normalize_letter
from openact_core.tasks.templates import ANSWER_PREFIXES
def _build_answer_prefix_regex() -> str:
    prefixes = sorted(set(ANSWER_PREFIXES.values()) | {"Answer"}, key=len, reverse=True)
    return "|".join(re.escape(p) for p in prefixes)
_ANSWER_PREFIX_RE = _build_answer_prefix_regex()
class MultipleChoiceParser(AnswerParser):
    def __init__(self, valid_letters: str = 'ABCDE'):
        self._valid: Set[str] = set(valid_letters.upper())
        self._letter_class = '[' + ''.join(sorted(self._valid)) + ']'
        self._patterns = self._build_patterns()
    def _build_patterns(self):
        lc = self._letter_class
        return [
            re.compile(
                rf'(?:^|\n)\s*(?:{_ANSWER_PREFIX_RE})\s*[:=：]\s*({lc})\b',
                re.IGNORECASE,
            ),
            re.compile(
                f'the\\s+answer\\s+is\\s*[:=]?\\s*({lc})\\b', re.IGNORECASE
            ),
            re.compile(
                f'\\(({lc})\\)', re.IGNORECASE
            ),
        ]
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        text = response.strip()
        for pattern in self._patterns:
            matches = list(pattern.finditer(text))
            if matches:
                letter = matches[-1].group(1).upper()
                if letter in self._valid:
                    return letter
        lc = self._letter_class
        line_start = re.compile(
            f'^({lc})\\s*[.)\\s]', re.MULTILINE | re.IGNORECASE
        )
        line_matches = list(line_start.finditer(text))
        if line_matches:
            letter = line_matches[-1].group(1).upper()
            if letter in self._valid:
                return letter
        fallback = re.compile(
            f'(?<![A-Za-z])({lc})(?![A-Za-z])', re.IGNORECASE
        )
        hits = list(fallback.finditer(text))
        if hits:
            return hits[-1].group(1).upper()
        return None
    def normalize(self, answer: Optional[str]) -> str:
        return normalize_letter(answer)
class MCParser4(MultipleChoiceParser):
    def __init__(self):
        super().__init__(valid_letters='ABCD')
class MCParser5(MultipleChoiceParser):
    def __init__(self):
        super().__init__(valid_letters='ABCDE')
