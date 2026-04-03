from abc import ABC, abstractmethod
from typing import Optional, List, Iterator
import re
import math
class AnswerParser(ABC):
    @abstractmethod
    def extract(self, response: str) -> Optional[str]:
        ...
    @abstractmethod
    def normalize(self, answer: Optional[str]) -> str:
        ...
    def extract_and_normalize(self, response: str) -> str:
        return self.normalize(self.extract(response))
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
def normalize_numeric(answer: Optional[str]) -> str:
    if answer is None:
        return ''
    s = str(answer).strip()
    s = s.lstrip('$').rstrip('%').strip()
    s = s.replace(',', '')
    s = re.sub(r'\.$', '', s)
    if not s:
        return ''
    try:
        num = float(s)
        if math.isnan(num) or math.isinf(num):
            return s
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return s
def normalize_letter(answer: Optional[str]) -> str:
    if answer is None:
        return ''
    s = str(answer).strip().upper()
    match = re.search(r'[A-E]', s)
    if match:
        return match.group(0)
    return ''
def normalize_text(answer: Optional[str]) -> str:
    if answer is None:
        return ''
    s = str(answer).strip()
    s = ' '.join(s.split())
    s = re.sub(r'[.,;:!?]+$', '', s)
    return s.lower()
def normalize_math_expr(answer: Optional[str]) -> str:
    if answer is None:
        return ''
    s = str(answer).strip()
    s = s.replace('$', '')
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    s = re.sub(r'\\(?:left|right|displaystyle|,|;|!|quad|qquad)\b', '', s)
    s = s.replace('\\dfrac', '\\frac')
    s = ' '.join(s.split())
    s = re.sub(r'[.,;:!?]+$', '', s)
    return s.lower()
_RE_NUMBER = r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?'
_RE_BOXED_LMD = re.compile(
    r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
    re.DOTALL,
)
def extract_boxed_regex(text: str) -> Optional[str]:
    matches = list(_RE_BOXED_LMD.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip() or None
def extract_first_boxed_regex(text: str) -> Optional[str]:
    m = _RE_BOXED_LMD.search(text)
    if not m:
        return None
    return m.group(1).strip() or None
def _iter_boxed_contents(text: str) -> Iterator[str]:
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        start = idx + 7
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            yield text[start:j - 1]
            i = j
        else:
            break
def extract_boxed(text: str) -> Optional[str]:
    last_content: Optional[str] = None
    for content in _iter_boxed_contents(text):
        last_content = content
    return last_content
def extract_first_boxed(text: str) -> Optional[str]:
    for content in _iter_boxed_contents(text):
        return content
    return None
def find_last_number(text: str) -> Optional[str]:
    numbers = re.findall(_RE_NUMBER, text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None
