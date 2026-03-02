"""
Code extraction parser for HumanEval and similar code-generation tasks.

Extracts Python code from model responses that may wrap it in Markdown
fences or produce it bare.
"""

import re
from typing import Optional

from openact_core.tasks.parsers.base import AnswerParser


class CodeParser(AnswerParser):
    """Extract code blocks from model responses.

    Extraction priority
    -------------------
    1. Fenced Python block  ````` ```python ... ``` `````
    2. Generic fenced block ````` ``` ... ``` `````
    3. Raw response text (stripped)

    Normalization strips leading/trailing whitespace only.
    """

    _RE_PYTHON_FENCE = re.compile(
        r"```python\s*\n(.*?)```", re.DOTALL
    )
    _RE_GENERIC_FENCE = re.compile(
        r"```\s*\n(.*?)```", re.DOTALL
    )

    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None

        # 1. ```python ... ```
        match = self._RE_PYTHON_FENCE.search(response)
        if match:
            return match.group(1).strip()

        # 2. ``` ... ```
        match = self._RE_GENERIC_FENCE.search(response)
        if match:
            return match.group(1).strip()

        # 3. Raw text
        return response.strip()

    def normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ""
        return answer.strip()
