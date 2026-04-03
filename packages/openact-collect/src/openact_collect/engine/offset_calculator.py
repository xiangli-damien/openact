import logging
import warnings
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
                warnings.warn(
                    f"Offset validation failed ({method_used}): "
                    f"{validation['errors'][:3]}",
                    UserWarning,
                )
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
        n_tokens = len(token_ids)
        offsets = np.zeros((n_tokens, 2), dtype=np.int32)
        content_ids = []
        content_map = []
        for i, tid in enumerate(token_ids):
            if tid not in self._special_ids:
                content_ids.append(tid)
                content_map.append(i)
        prev_decoded = ""
        char_pos = 0
        content_offsets = np.zeros((len(content_ids), 2), dtype=np.int32)
        for ci in range(len(content_ids)):
            current_decoded = self.tokenizer.decode(
                content_ids[: ci + 1],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            new_text = current_decoded[len(prev_decoded) :]
            new_len = len(new_text)
            if new_len > 0:
                search_start = max(0, char_pos - max_search)
                search_end = min(len(text), char_pos + max_search + new_len)
                found_pos = text.find(new_text, search_start, search_end)
                if found_pos != -1:
                    content_offsets[ci, 0] = found_pos
                    content_offsets[ci, 1] = found_pos + new_len
                    char_pos = found_pos + new_len
                else:
                    content_offsets[ci, 0] = char_pos
                    content_offsets[ci, 1] = min(char_pos + new_len, len(text))
                    char_pos = content_offsets[ci, 1]
            else:
                content_offsets[ci, 0] = char_pos
                content_offsets[ci, 1] = char_pos
            prev_decoded = current_decoded
        ci = 0
        prev_end = 0
        for i in range(n_tokens):
            if token_ids[i] in self._special_ids:
                offsets[i, 0] = prev_end
                offsets[i, 1] = prev_end
            else:
                if ci < len(content_offsets):
                    offsets[i, 0] = content_offsets[ci, 0]
                    offsets[i, 1] = content_offsets[ci, 1]
                    prev_end = content_offsets[ci, 1]
                    ci += 1
                else:
                    offsets[i, 0] = prev_end
                    offsets[i, 1] = prev_end
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
