import re
import math
import ast
import warnings
from typing import Optional
from openact_core.tasks.parsers.base import (
    AnswerParser,
    normalize_numeric,
    normalize_letter,
    normalize_text,
    extract_first_boxed,
)
_RE_NUMBER = re.compile(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?')
class TheoremQAParser(AnswerParser):
    _RE_ANSWER_MARKER = re.compile(r'[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)')
    _RE_THEREFORE = re.compile(
        r'[Tt]herefore,?\s*(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.(?!\d)|\n|$)',
        re.IGNORECASE,
    )
    _RE_THEREFORE_BROAD = re.compile(
        r'[Tt]herefore,?\s+.+?\s+is\s+(.+?)(?:\.(?!\d)|\n|$)',
        re.IGNORECASE,
    )
    _RE_FRAC = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
    _RE_OPTION = re.compile(r'\(([a-dA-D])\)')
    _RE_PI_COEFF = re.compile(r'(\d+(?:\.\d+)?)\s*(?:\\)?pi\b')
    def __init__(self, answer_type: str = 'float'):
        self.answer_type = answer_type.lower().strip()
    def extract(self, response: str) -> Optional[str]:
        if not response:
            return None
        boxed = extract_first_boxed(response)
        if boxed is not None:
            return boxed.strip()
        match = self._RE_THEREFORE.search(response)
        if match:
            return match.group(1).strip()
        match = self._RE_ANSWER_MARKER.search(response)
        if match:
            return match.group(1).strip()
        match = self._RE_THEREFORE_BROAD.search(response)
        if match:
            return match.group(1).strip()
        if self.answer_type in ('float', 'integer', 'int'):
            nums = _RE_NUMBER.findall(response)
            if nums:
                return nums[-1].replace(',', '')
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line
        return None
    def normalize(self, answer: Optional[str]) -> str:
        if answer is None:
            return ''
        answer = str(answer).strip()
        answer = self._clean_latex(answer)
        if self.answer_type == 'bool':
            return self._normalize_bool(answer)
        elif self.answer_type == 'option':
            return self._normalize_option(answer)
        elif self.answer_type in ('integer', 'int'):
            return self._normalize_integer(answer)
        elif self.answer_type == 'float':
            return self._normalize_float(answer)
        elif 'list' in self.answer_type:
            return self._normalize_list(answer)
        else:
            return normalize_text(answer)
    def _clean_latex(self, text: str) -> str:
        text = text.replace('$', '')
        frac_match = self._RE_FRAC.search(text)
        if frac_match:
            try:
                num = float(frac_match.group(1))
                den = float(frac_match.group(2))
                if den != 0:
                    text = text.replace(frac_match.group(0), str(num / den))
            except (ValueError, ZeroDivisionError):
                pass
        text = self._RE_PI_COEFF.sub(lambda m: str(float(m.group(1)) * math.pi), text)
        text = re.sub(r'\\pi\b', str(math.pi), text)
        text = re.sub(r'\bpi\b', str(math.pi), text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = text.replace('\\(', '').replace('\\)', '')
        text = text.replace('{', '').replace('}', '')
        stripped = text.replace(' ', '')
        stripped = re.sub(r'(?<=\d)\(', '*(', stripped)
        stripped = re.sub(r'\)(?=\d)', ')*', stripped)
        if all(c in '0123456789.*()+-eE' for c in stripped):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', SyntaxWarning)
                    result = eval(compile(stripped, '<expr>', 'eval'), {'__builtins__': {}}, {})
                if isinstance(result, (int, float)):
                    text = str(result)
            except Exception:
                pass
        return text.strip()
    @staticmethod
    def _try_extract_number(text: str) -> Optional[str]:
        nums = _RE_NUMBER.findall(text)
        if nums:
            return nums[-1].replace(',', '')
        return None
    @staticmethod
    def _normalize_bool(answer: str) -> str:
        low = answer.lower().strip()
        if low in ('true', 'yes', '1', 't'):
            return 'true'
        if low in ('false', 'no', '0', 'f'):
            return 'false'
        return low
    def _normalize_option(self, answer: str) -> str:
        m = self._RE_OPTION.search(answer)
        if m:
            return m.group(1).upper()
        return normalize_letter(answer)
    @staticmethod
    def _normalize_integer(answer: str) -> str:
        s = normalize_numeric(answer)
        if s:
            try:
                float(s)
                return s
            except ValueError:
                pass
        num = TheoremQAParser._try_extract_number(answer)
        if num:
            return normalize_numeric(num)
        return s
    @staticmethod
    def _normalize_float(answer: str) -> str:
        s = normalize_numeric(answer)
        if not s:
            return ''
        try:
            val = float(s)
            return f'{val:.6f}'.rstrip('0').rstrip('.')
        except ValueError:
            pass
        num = TheoremQAParser._try_extract_number(answer)
        if num:
            try:
                val = float(num)
                return f'{val:.6f}'.rstrip('0').rstrip('.')
            except ValueError:
                pass
        return s
    @staticmethod
    def _normalize_list(answer: str) -> str:
        nums = TheoremQAParser._parse_list(answer)
        if nums is None:
            return answer.strip().lower()
        normalized = []
        for n in nums:
            if float(n).is_integer():
                normalized.append(str(int(n)))
            else:
                normalized.append(str(float(n)))
        return '[' + ', '.join(normalized) + ']'
    @staticmethod
    def _parse_list(text: str) -> Optional[list]:
        text = str(text).strip()
        try:
            result = ast.literal_eval(text)
            if isinstance(result, (list, tuple)):
                return [float(x) for x in result]
        except (ValueError, SyntaxError):
            pass
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if nums:
            return [float(n) for n in nums]
        return None
