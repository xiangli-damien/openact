import re
from typing import Optional, Set
from openact_core.tasks.parsers.base import AnswerParser, normalize_letter


class MultipleChoiceParser(AnswerParser):

    def __init__(self, valid_letters: str = 'ABCDE'):
        self._valid: Set[str] = set(valid_letters.upper())
        self._letter_class = '[' + ''.join(sorted(self._valid)) + ']'
        self._patterns = self._build_patterns()

    def _build_patterns(self):
        lc = self._letter_class
        # All patterns use re.IGNORECASE so they work regardless of whether the
        # input has been uppercased or not.
        return [
            re.compile(
                f'answer\\s*[:=]\\s*({lc})\\b', re.IGNORECASE
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
        # First pass: try structured answer-marker patterns on the ORIGINAL
        # text (case-insensitive).  This ensures "Answer: B" at the end wins
        # over an "A." that appears as a choice label earlier in the text.
        for pattern in self._patterns:
            # Search from the end by taking the last match
            matches = list(pattern.finditer(text))
            if matches:
                letter = matches[-1].group(1).upper()
                if letter in self._valid:
                    return letter

        # Second pass: look for a bare letter at the start of a line
        # e.g. "B. Second option" — but only if it is the LAST such
        # occurrence, to prefer a final answer over choice labels.
        lc = self._letter_class
        line_start = re.compile(
            f'^({lc})\\s*[.)\\s]', re.MULTILINE | re.IGNORECASE
        )
        line_matches = list(line_start.finditer(text))
        if line_matches:
            letter = line_matches[-1].group(1).upper()
            if letter in self._valid:
                return letter

        # Final fallback: find the last isolated valid letter in the text
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