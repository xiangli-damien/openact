import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
logger = logging.getLogger(__name__)
_MAX_SEARCH_DISTANCE = 200
class OffsetCalculator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        special_ids: Optional[Set[int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._has_fast_tokenizer = getattr(tokenizer, "is_fast", False)
        self._special_ids = special_ids or set(
            getattr(tokenizer, "all_special_ids", []) or []
        )
    def compute_offsets(
        self, text: str, token_ids: List[int], validate: bool = True
    ) -> np.ndarray:
        if not token_ids:
            return np.zeros((0, 2), dtype=np.int32)
        offsets = None
        method_used = None
        if self._has_fast_tokenizer:
            try:
                offsets = self._method_native_offset_mapping(text, token_ids)
                if offsets is not None:
                    method_used = "native_offset_mapping"
            except Exception:
                logger.debug(
                    "native_offset_mapping failed, will try incremental_decode",
                    exc_info=True,
                )
        if offsets is None:
            try:
                offsets = self._method_incremental_decode(
                    text, token_ids, _MAX_SEARCH_DISTANCE
                )
                if offsets is not None:
                    method_used = "incremental_decode"
            except Exception:
                logger.debug(
                    "incremental_decode failed for offset calculation",
                    exc_info=True,
                )
        if offsets is None:
            raise RuntimeError(
                f"All offset calculation methods failed for tokenizer "
                f"{type(self.tokenizer).__name__}. "
                f"Text length: {len(text)}, token count: {len(token_ids)}. "
                f"Please report this issue."
            )
        if validate and offsets is not None:
            validation = self._validate_offsets(text, token_ids, offsets)
            if not validation["is_valid"]:
                raise ValueError(f"Offset validation failed ({method_used}): {validation['errors'][:3]}")
        return offsets
    def _method_native_offset_mapping(
        self, text: str, token_ids: List[int]
    ) -> Optional[np.ndarray]:
        content_ids = [tid for tid in token_ids if tid not in self._special_ids]
        is_special = [tid in self._special_ids for tid in token_ids]
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None,
        )
        encoded_ids = encoding.get("input_ids", [])
        if list(encoded_ids) != list(content_ids):
            return None
        offset_mapping = encoding.get("offset_mapping")
        if offset_mapping is None:
            return None
        content_offsets = list(offset_mapping)
        offsets = np.zeros((len(token_ids), 2), dtype=np.int32)
        content_idx = 0
        prev_end = 0
        for i in range(len(token_ids)):
            if is_special[i]:
                offsets[i, 0] = prev_end
                offsets[i, 1] = prev_end
            else:
                if content_idx < len(content_offsets):
                    s, e = content_offsets[content_idx]
                    if s == 0 and e == 0 and content_idx > 0:
                        s = prev_end
                        e = prev_end
                    offsets[i, 0] = s
                    offsets[i, 1] = e
                    prev_end = e
                    content_idx += 1
                else:
                    offsets[i, 0] = prev_end
                    offsets[i, 1] = prev_end
        return offsets
    def _method_incremental_decode(
        self, text: str, token_ids: List[int], max_search: int
    ) -> Optional[np.ndarray]:
        del max_search
        # Re-tokenization may change generated token boundaries. Decode actual
        # prefixes, matching only from the start; substring search corrupts
        # repeated text and incomplete UTF-8 tokens.
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ValueError('Response text does not match the stored token IDs')
        offsets = np.zeros((len(token_ids), 2), dtype=np.int32)
        previous_end = 0
        pending = []
        for index, token_id in enumerate(token_ids):
            if token_id in self._special_ids:
                offsets[index] = (previous_end, previous_end)
                continue
            pending.append(index)
            prefix = self.tokenizer.decode(token_ids[:index + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            end = 0
            for actual, expected in zip(prefix, text):
                if actual != expected:
                    break
                end += 1
            # SentencePiece can temporarily replace an entire byte run with
            # replacement characters when its next code point is incomplete.
            # Keep the prefix already verified against the final decoded text.
            end = max(previous_end, end)
            if end > previous_end:
                # Several byte tokens can jointly encode a single character.
                for pending_index in pending:
                    offsets[pending_index] = (previous_end, end)
                pending.clear()
                previous_end = end
            else:
                offsets[index] = (previous_end, previous_end)
        return offsets

    def _validate_offsets(
        self, text: str, token_ids: List[int], offsets: np.ndarray
    ) -> Dict[str, Any]:
        errors = []
        n_tokens = len(token_ids)
        for i in range(1, n_tokens):
            if offsets[i, 0] < offsets[i - 1, 0]:
                errors.append(f"Token {i}: non-monotonic start")
        for i in range(n_tokens):
            if offsets[i, 0] > offsets[i, 1]:
                errors.append(f"Token {i}: start > end")
        for i in range(n_tokens):
            if offsets[i, 1] > len(text):
                errors.append(f"Token {i}: exceeds text length")
        content_indices = [
            i for i in range(n_tokens) if token_ids[i] not in self._special_ids
        ]
        if content_indices and len(text) > 0:
            last_content = content_indices[-1]
            coverage = offsets[last_content, 1] / len(text)
            if coverage < 0.9:
                errors.append(f"Low coverage: {coverage:.2%}")
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "n_errors": len(errors),
        }
    def validate_offsets(
        self, text: str, token_ids: List[int], offsets: np.ndarray
    ) -> Tuple[bool, List[str]]:
        result = self._validate_offsets(text, token_ids, offsets)
        return result["is_valid"], result["errors"]
