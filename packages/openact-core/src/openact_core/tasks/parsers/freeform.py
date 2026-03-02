"""
Freeform text answer parsers for tasks without a rigid answer format.

TruthfulQAParser  — extracts the answer line from a response.
IFEvalParser      — returns the full response (IFEval evaluation is
                    instruction-following, not answer-matching).
"""

import re
from typing import Optional

from openact_core.tasks.parsers.base import AnswerParser, normalize_text


class TruthfulQAParser(AnswerParser):
    """Extract the answer from a TruthfulQA response.

    Extraction priority
    -------------------
    1. ``Answer: <text>``              (explicit marker)
    2. ``the answer is <text>``        (natural-language marker)
    3. Last non-empty line of response (fallback)

    Normalization is light: lowercase, collapse whitespace, strip trailing
    punctuation.  Real TruthfulQA evaluation typically uses a trained judge
    model; this parser provides a structural extraction layer.
    """

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

        # 1. "Answer: ..."
        match = self._RE_ANSWER_MARKER.search(response)
        if match:
            return match.group(1).strip()

        # 2. "the answer is ..."
        match = self._RE_ANSWER_IS.search(response)
        if match:
            return match.group(1).strip()

        # 3. Last non-empty line
        lines = response.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line

        return None

    def normalize(self, answer: Optional[str]) -> str:
        return normalize_text(answer)


class IFEvalParser(AnswerParser):
    """Parser for IFEval (instruction-following evaluation).

    IFEval does not have traditional "answers" — correctness is determined by
    whether the response satisfies a set of verifiable instructions (e.g.
    word count, keyword inclusion, format constraints).  This parser simply
    returns the full response text, serving as a structural extraction layer
    for downstream IFEval evaluators.
    """

    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        return response.strip()

    def normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ""
        return answer.strip()
