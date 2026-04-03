import math
import re
from typing import Optional

from openact_eval.matchers.base import Matcher

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:
    sp = None
    parse_expr = None
    standard_transformations = None
    implicit_multiplication_application = None
    convert_xor = None


class MathMatcher(Matcher):
    def __init__(self, rel_tol: float = 0.001, abs_tol: float = 1e-6):
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)

    def match(self, extracted: str, ground_truth: str) -> bool:
        e_raw = "" if extracted is None else str(extracted)
        g_raw = "" if ground_truth is None else str(ground_truth)

        e = self._normalize_text(e_raw)
        g = self._normalize_text(g_raw)

        if not e or not g:
            return False
        if e == g:
            return True

        e_num = self._to_float(e)
        g_num = self._to_float(g)
        if e_num is not None and g_num is not None:
            if self.rel_tol == 0:
                return abs(e_num - g_num) <= self.abs_tol
            return math.isclose(e_num, g_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

        if sp is not None and parse_expr is not None:
            e_expr = self._to_sympy(e_raw)
            g_expr = self._to_sympy(g_raw)
            if e_expr is not None and g_expr is not None:
                try:
                    diff = sp.simplify(e_expr - g_expr)
                    if diff == 0:
                        return True
                except Exception:
                    pass

        return False

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        s = re.sub(r"^[Aa]nswer\s*:\s*", "", s).strip()
        s = s.replace("$", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\,", "").replace("\\!", "")
        s = s.strip()
        if s.endswith("."):
            s = s[:-1].strip()
        return s

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            pass
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$", s)
        if m:
            try:
                num = float(m.group(1))
                den = float(m.group(2))
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None
        return None

    def _to_sympy(self, s: str):
        if sp is None or parse_expr is None:
            return None
        expr_str = self._latex_to_expr_string(s)
        if not expr_str:
            return None
        try:
            local_dict = {"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt}
            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )
            return parse_expr(
                expr_str,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            )
        except Exception:
            return None

    @classmethod
    def _latex_to_expr_string(cls, s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        s = re.sub(r"^[Aa]nswer\s*:\s*", "", s).strip()
        s = s.replace("$", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\,", "").replace("\\!", "")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("\\pi", "pi")
        s = cls._strip_wrapped_tex_command(s, "boxed")
        s = cls._strip_wrapped_tex_command(s, "text")
        s = cls._strip_wrapped_tex_command(s, "mathrm")
        s = cls._strip_wrapped_tex_command(s, "mathbf")
        s = cls._replace_tex_frac(s)
        s = cls._replace_tex_sqrt(s)
        s = s.replace("{", "(").replace("}", ")")
        s = s.strip()
        if s.endswith("."):
            s = s[:-1].strip()
        return s

    @staticmethod
    def _strip_wrapped_tex_command(s: str, name: str) -> str:
        prefix = f"\\{name}"
        i = 0
        out = ""
        while i < len(s):
            if s.startswith(prefix, i):
                j = i + len(prefix)
                while j < len(s) and s[j].isspace():
                    j += 1
                if j < len(s) and s[j] == "{":
                    content, k = MathMatcher._extract_braced(s, j)
                    out += content
                    i = k
                    continue
            out += s[i]
            i += 1
        return out

    @staticmethod
    def _replace_tex_frac(s: str) -> str:
        for cmd in ("\\frac", "\\dfrac", "\\tfrac"):
            s = MathMatcher._replace_two_arg_command(
                s, cmd, lambda a, b: f"(({a})/({b}))"
            )
        return s

    @staticmethod
    def _replace_tex_sqrt(s: str) -> str:
        return MathMatcher._replace_one_arg_command(s, "\\sqrt", lambda a: f"sqrt({a})")

    @staticmethod
    def _replace_two_arg_command(s: str, cmd: str, fmt):
        i = 0
        out = ""
        while i < len(s):
            if s.startswith(cmd, i):
                j = i + len(cmd)
                while j < len(s) and s[j].isspace():
                    j += 1
                if j < len(s) and s[j] == "{":
                    a, k = MathMatcher._extract_braced(s, j)
                    j2 = k
                    while j2 < len(s) and s[j2].isspace():
                        j2 += 1
                    if j2 < len(s) and s[j2] == "{":
                        b, k2 = MathMatcher._extract_braced(s, j2)
                        out += fmt(a, b)
                        i = k2
                        continue
            out += s[i]
            i += 1
        return out

    @staticmethod
    def _replace_one_arg_command(s: str, cmd: str, fmt):
        i = 0
        out = ""
        while i < len(s):
            if s.startswith(cmd, i):
                j = i + len(cmd)
                while j < len(s) and s[j].isspace():
                    j += 1
                if j < len(s) and s[j] == "{":
                    a, k = MathMatcher._extract_braced(s, j)
                    out += fmt(a)
                    i = k
                    continue
            out += s[i]
            i += 1
        return out

    @staticmethod
    def _extract_braced(s: str, start: int):
        if start >= len(s) or s[start] != "{":
            return "", start
        depth = 1
        i = start + 1
        buf = []
        while i < len(s) and depth > 0:
            c = s[i]
            if c == "{":
                depth += 1
                buf.append(c)
            elif c == "}":
                depth -= 1
                if depth > 0:
                    buf.append(c)
            else:
                buf.append(c)
            i += 1
        return "".join(buf), i