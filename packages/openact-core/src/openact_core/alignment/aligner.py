"""
Character ↔ token alignment utilities.

:class:`TokenAligner` maps between character offsets in a text string and the
token indices produced by a tokenizer.  It is the backbone of substring-based
hidden-state retrieval in :class:`openact_core.io.sample.Sample`.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Iterator
import bisect
import numpy as np


@dataclass(frozen=True)
class CharSpan:
    """Half-open character span ``[start, end)``."""

    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def __bool__(self) -> bool:
        return self.end > self.start

    def overlaps(self, other: "CharSpan") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, char_idx: int) -> bool:
        return self.start <= char_idx < self.end

    def to_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True)
class TokenSpan:
    """Half-open token span ``[start, end)``."""

    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def __bool__(self) -> bool:
        return self.end > self.start

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end))

    def to_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)

    def to_slice(self) -> slice:
        return slice(self.start, self.end)


class TokenAligner:
    """Bidirectional map between character positions and token indices.

    Parameters
    ----------
    text : str
        The full text that was tokenized.
    offsets : array-like
        Shape ``(n_tokens, 2)`` array of ``(char_start, char_end)`` per token.
    """

    def __init__(self, text: str, offsets: np.ndarray):
        self.text = text
        self.offsets = np.asarray(offsets, dtype=np.int32)
        self.n_tokens = len(offsets)
        if self.n_tokens > 0:
            self._char_starts = self.offsets[:, 0].copy()
            self._char_ends = self.offsets[:, 1].copy()
        else:
            self._char_starts = np.array([], dtype=np.int32)
            self._char_ends = np.array([], dtype=np.int32)

    # -- Token → Character ---------------------------------------------------

    def token_to_char_span(self, token_idx: int) -> CharSpan:
        """Return the character span for a single token index."""
        if token_idx < 0:
            token_idx = self.n_tokens + token_idx
        if token_idx < 0 or token_idx >= self.n_tokens:
            raise IndexError(
                f"Token index {token_idx} out of range [0, {self.n_tokens})"
            )
        return CharSpan(
            start=int(self.offsets[token_idx, 0]),
            end=int(self.offsets[token_idx, 1]),
        )

    def token_to_text(self, token_idx: int) -> str:
        """Return the substring corresponding to *token_idx*."""
        span = self.token_to_char_span(token_idx)
        return self.text[span.start : span.end]

    def token_span_to_char_span(self, token_span: TokenSpan) -> CharSpan:
        """Return the character span covering all tokens in *token_span*."""
        if not token_span or token_span.start >= token_span.end:
            return CharSpan(0, 0)
        if token_span.start >= self.n_tokens:
            return CharSpan(0, 0)
        end_idx = min(token_span.end - 1, self.n_tokens - 1)
        return CharSpan(
            start=int(self.offsets[token_span.start, 0]),
            end=int(self.offsets[end_idx, 1]),
        )

    def token_span_to_text(self, token_span: TokenSpan) -> str:
        """Return the text covered by *token_span*."""
        char_span = self.token_span_to_char_span(token_span)
        return self.text[char_span.start : char_span.end]

    def tokens_to_texts(
        self, start: int = 0, end: Optional[int] = None
    ) -> List[str]:
        """Return per-token text for tokens in ``[start, end)``."""
        if end is None:
            end = self.n_tokens
        return [self.token_to_text(i) for i in range(start, end)]

    # -- Character → Token ---------------------------------------------------

    def char_to_token(self, char_idx: int) -> Optional[int]:
        """Return the token index containing *char_idx*, or ``None``."""
        if self.n_tokens == 0:
            return None
        token_idx = bisect.bisect_right(self._char_starts, char_idx) - 1
        if token_idx < 0:
            return None
        if char_idx < self._char_ends[token_idx]:
            return token_idx
        return None

    def char_span_to_token_span(self, char_span: CharSpan) -> TokenSpan:
        """Return the minimal token span that covers *char_span*."""
        if not char_span or char_span.start >= char_span.end:
            return TokenSpan(0, 0)
        if self.n_tokens == 0:
            return TokenSpan(0, 0)
        start_tok = bisect.bisect_right(self._char_ends, char_span.start)
        end_tok = bisect.bisect_left(self._char_starts, char_span.end)
        start_tok = max(0, start_tok)
        end_tok = min(self.n_tokens, end_tok)
        if start_tok >= end_tok:
            return TokenSpan(0, 0)
        return TokenSpan(start=start_tok, end=end_tok)

    # -- Substring search ----------------------------------------------------

    def find_substring(
        self, substring: str, occurrence: int = -1
    ) -> Optional[TokenSpan]:
        """Find a substring and return its token span.

        Parameters
        ----------
        substring : str
            The text to search for.
        occurrence : int
            Which occurrence to return.  ``-1`` (default) returns the *last*
            occurrence, which is usually the right choice when extracting a
            model's final answer.  ``0`` returns the first, ``1`` the second,
            etc.
        """
        if not substring:
            return None
        positions = []
        start = 0
        while True:
            pos = self.text.find(substring, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        if not positions:
            return None
        if occurrence == -1:
            char_start = positions[-1]
        elif 0 <= occurrence < len(positions):
            char_start = positions[occurrence]
        else:
            return None
        char_span = CharSpan(start=char_start, end=char_start + len(substring))
        return self.char_span_to_token_span(char_span)

    def find_last(self, substring: str) -> Optional[TokenSpan]:
        """Convenience shortcut: find the *last* occurrence of *substring*.

        Equivalent to ``find_substring(substring, occurrence=-1)``.  This is
        the most common use-case (locating a model's final answer).
        """
        return self.find_substring(substring, occurrence=-1)

    def find_first(self, substring: str) -> Optional[TokenSpan]:
        """Convenience shortcut: find the *first* occurrence of *substring*.

        Equivalent to ``find_substring(substring, occurrence=0)``.
        """
        return self.find_substring(substring, occurrence=0)

    def find_all_substrings(self, substring: str) -> List[TokenSpan]:
        """Return token spans for every non-overlapping occurrence."""
        if not substring:
            return []
        spans = []
        start = 0
        while True:
            pos = self.text.find(substring, start)
            if pos == -1:
                break
            char_span = CharSpan(start=pos, end=pos + len(substring))
            token_span = self.char_span_to_token_span(char_span)
            if token_span:
                spans.append(token_span)
            start = pos + 1
        return spans

    # -- Inspection / debugging ----------------------------------------------

    def get_token_boundaries(self) -> List[Tuple[int, int, str]]:
        """Return ``(char_start, char_end, text)`` for every token."""
        return [
            (int(self.offsets[i, 0]), int(self.offsets[i, 1]), self.token_to_text(i))
            for i in range(self.n_tokens)
        ]

    def visualize(
        self,
        highlight_span: Optional[TokenSpan] = None,
        max_tokens: int = 50,
    ) -> str:
        """Return a compact string visualization of the token sequence."""
        parts = []
        n_show = min(self.n_tokens, max_tokens)
        for i in range(n_show):
            text = self.token_to_text(i)
            text = text.replace("\n", "\\n").replace("\t", "\\t")
            if highlight_span and highlight_span.start <= i < highlight_span.end:
                parts.append(f"[{text}]")
            else:
                parts.append(f"|{text}")
        result = "".join(parts) + "|"
        if self.n_tokens > max_tokens:
            result += f" ... ({self.n_tokens - max_tokens} more tokens)"
        return result

    def validate(self) -> List[str]:
        """Check internal consistency; return a list of issues found."""
        issues = []
        if self.n_tokens == 0:
            return issues
        for i in range(1, self.n_tokens):
            if self._char_starts[i] < self._char_starts[i - 1]:
                issues.append(
                    f"Non-monotonic char_starts at token {i}: "
                    f"{self._char_starts[i]} < {self._char_starts[i - 1]}"
                )
        for i in range(self.n_tokens):
            if self._char_starts[i] > self._char_ends[i]:
                issues.append(
                    f"Invalid span at token {i}: "
                    f"start {self._char_starts[i]} > end {self._char_ends[i]}"
                )
        if len(self.text) > 0 and self._char_ends[-1] > len(self.text):
            issues.append(
                f"Offsets exceed text length: "
                f"max offset {self._char_ends[-1]} > text length {len(self.text)}"
            )
        return issues

    def __len__(self) -> int:
        return self.n_tokens

    def __repr__(self) -> str:
        return f"TokenAligner(n_tokens={self.n_tokens}, text_len={len(self.text)})"