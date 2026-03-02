"""
Numeric answer parser for tasks whose ground truth is a number.

Covers GSM8K and any other task where the expected answer is an integer or
decimal.  The extraction cascade tries several common formats in priority
order so that even poorly-formatted model outputs are handled gracefully.
"""

import re
from typing import Optional

from openact_core.tasks.parsers.base import (
    AnswerParser,
    normalize_numeric,
    extract_boxed,
    find_last_number,
    _RE_NUMBER,
)


class NumericParser(AnswerParser):
    """Extract and normalize numeric answers.

    Extraction priority
    -------------------
    1. ``Answer: <number>``            (explicit marker)
    2. ``the answer is <number>``      (natural-language marker)
    3. ``#### <number>``               (GSM8K gold format)
    4. ``\\boxed{<number>}``           (LaTeX)
    5. Last number in response         (fallback)

    All steps are case-insensitive except the LaTeX pattern.
    """

    # Pre-compiled patterns ordered by specificity.
    _PATTERNS = [
        # 1. "Answer: 42" or "answer: -3.14"
        re.compile(
            r"[Aa]nswer\s*[:=]\s*\$?\s*(" + _RE_NUMBER + r")",
        ),
        # 2. "the answer is 42"
        re.compile(
            r"[Tt]he\s+answer\s+is\s*[:=]?\s*\$?\s*(" + _RE_NUMBER + r")",
        ),
        # 3. "#### 42"
        re.compile(
            r"####\s*(" + _RE_NUMBER + r")",
        ),
    ]

    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None

        # Try structured patterns first.
        for pattern in self._PATTERNS:
            match = pattern.search(response)
            if match:
                return match.group(1).replace(",", "")

        # LaTeX \boxed{...}
        boxed = extract_boxed(response)
        if boxed is not None:
            # The boxed content might itself be an expression; try to pull a
            # number out of it.
            num = find_last_number(boxed)
            if num is not None:
                return num
            return boxed.strip()

        # Ultimate fallback: last number in text.
        return find_last_number(response)

    def normalize(self, answer: Optional[str]) -> str:
        return normalize_numeric(answer)
