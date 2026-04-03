import re
from typing import Optional
from openact_core.tasks.parsers.base import AnswerParser
class CodeParser(AnswerParser):
    _RE_PYTHON_FENCE = re.compile(
        r"```python\s*\n(.*?)```", re.DOTALL
    )
    _RE_GENERIC_FENCE = re.compile(
        r"```\s*\n(.*?)```", re.DOTALL
    )
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        match = self._RE_PYTHON_FENCE.search(response)
        if match:
            return match.group(1).strip()
        match = self._RE_GENERIC_FENCE.search(response)
        if match:
            return match.group(1).strip()
        return response.strip()
    def normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ""
        return answer.strip()
