from abc import ABC, abstractmethod
from typing import Optional, List
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
    s = re.sub('\\.$', '', s)
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
    match = re.search('[A-E]', s)
    if match:
        return match.group(0)
    return ''


def normalize_text(answer: Optional[str]) -> str:
    if answer is None:
        return ''
    s = str(answer).strip()
    s = ' '.join(s.split())
    s = re.sub('[.,;:!?]+$', '', s)
    return s.lower()


def normalize_math_expr(answer: Optional[str]) -> str:
    if answer is None:
        return ''
    s = str(answer).strip()
    s = s.replace('$', '')
    s = re.sub('\\\\text\\{([^}]*)\\}', '\\1', s)
    s = re.sub(
        '\\\\(?:left|right|displaystyle|,|;|!|quad|qquad)\\b', '', s
    )
    s = s.replace('\\frac', 'frac')
    s = ' '.join(s.split())
    s = re.sub('[.,;:!?]+$', '', s)
    return s.lower()


_RE_NUMBER = '[+-]?\\d+(?:,\\d{3})*(?:\\.\\d+)?'


def extract_boxed(text: str) -> Optional[str]:
    last_content: Optional[str] = None
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
            last_content = text[start : j - 1]
        i = idx + 7
    return last_content


def find_last_number(text: str) -> Optional[str]:
    numbers = re.findall(_RE_NUMBER, text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None