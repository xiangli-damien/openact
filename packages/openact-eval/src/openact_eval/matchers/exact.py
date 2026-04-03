from openact_eval.matchers.base import Matcher
class ExactMatcher(Matcher):
    def match(self, extracted: str, ground_truth: str) -> bool:
        return self._normalize(extracted) == self._normalize(ground_truth)
    @staticmethod
    def _normalize(s: str) -> str:
        s = str(s).strip()
        if s.endswith("."):
            s = s[:-1].strip()
        return s
