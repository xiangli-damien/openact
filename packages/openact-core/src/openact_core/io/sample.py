from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from openact_core.alignment.aligner import CharSpan, TokenAligner, TokenSpan
from openact_core.schema.status import SampleStatus

if TYPE_CHECKING:
    from openact_core.io.run import Run


class Sample:
    def __init__(self, run: "Run", idx: int):
        self._run = run
        self._idx = idx
        self._aligner_cache: Optional[TokenAligner] = None
        self._meta_cache: Optional[Dict[str, Any]] = None
        self._hidden_states_cache: Optional[np.ndarray] = None
        self._ptr_cache: Optional[Tuple[int, int]] = None

    @property
    def index(self) -> int:
        return self._idx

    @property
    def sample_idx(self) -> int:
        return self._idx

    @property
    def sample_id(self) -> str:
        return str(self.meta.get("sample_id", self._idx))

    @property
    def status(self) -> SampleStatus:
        return SampleStatus(int(self._run._df.iloc[self._idx]["status"]))

    @property
    def is_valid(self) -> bool:
        return self.status == SampleStatus.OK

    @property
    def prompt_text(self) -> str:
        return str(self.meta.get("prompt_text", ""))

    @property
    def response_text(self) -> str:
        return str(self.meta.get("response_text", ""))

    @property
    def full_text(self) -> str:
        return self.prompt_text + self.response_text

    @property
    def ground_truth(self) -> Optional[str]:
        value = self.meta.get("ground_truth")
        return None if value is None else str(value)

    @property
    def finish_reason(self) -> str:
        return str(self.meta.get("finish_reason", "unknown"))

    @property
    def _ptr(self) -> Tuple[int, int]:
        if self._ptr_cache is None:
            ptr = self._run._zarr["tokens/sample_ptr"]
            self._ptr_cache = (int(ptr[self._idx]), int(ptr[self._idx + 1]))
        return self._ptr_cache

    @property
    def n_tokens(self) -> int:
        start, end = self._ptr
        return end - start

    @property
    def token_ids(self) -> np.ndarray:
        start, end = self._ptr
        return np.asarray(self._run._zarr["tokens/ids"][start:end])

    @property
    def token_offsets(self) -> np.ndarray:
        start, end = self._ptr
        return np.asarray(self._run._zarr["tokens/offsets"][start:end])

    @property
    def aligner(self) -> TokenAligner:
        if self._aligner_cache is None:
            self._aligner_cache = TokenAligner(self.response_text, self.token_offsets)
        return self._aligner_cache

    def token_to_text(self, token_idx: int) -> str:
        return self.aligner.token_to_text(token_idx)

    def span_to_text(self, span: TokenSpan) -> str:
        return self.aligner.token_span_to_text(span)

    def find_text(self, substring: str, occurrence: int = -1) -> Optional[TokenSpan]:
        return self.aligner.find_substring(substring, occurrence)

    def find_all_text(self, substring: str) -> List[TokenSpan]:
        return self.aligner.find_all_substrings(substring)

    @property
    def hidden_states(self) -> np.ndarray:
        if self._hidden_states_cache is not None:
            return self._hidden_states_cache

        start, end = self._ptr
        if "hidden_states/per_token" in self._run._zarr:
            self._hidden_states_cache = np.asarray(self._run._zarr["hidden_states/per_token"][start:end])
            return self._hidden_states_cache
        if "response/hidden_states/data" in self._run._zarr:
            self._hidden_states_cache = np.asarray(self._run._zarr["response/hidden_states/data"][start:end])
            return self._hidden_states_cache
        if "hidden_states/mean" in self._run._zarr:
            self._hidden_states_cache = np.asarray(self._run._zarr["hidden_states/mean"][self._idx])[None, ...]
            return self._hidden_states_cache

        self._hidden_states_cache = np.array([])
        return self._hidden_states_cache

    @property
    def mean_hidden_states(self) -> np.ndarray:
        if "hidden_states/mean" in self._run._zarr:
            return np.asarray(self._run._zarr["hidden_states/mean"][self._idx])
        hs = self.hidden_states
        return np.array([]) if len(hs) == 0 else hs.mean(axis=0).astype(np.float32)

    @property
    def prompt_last_hidden_states(self) -> np.ndarray:
        if "hidden_states/prompt_last" in self._run._zarr:
            return np.asarray(self._run._zarr["hidden_states/prompt_last"][self._idx])
        return np.array([])

    @property
    def generated_only_hidden_states(self) -> np.ndarray:
        return self.hidden_states

    def get_layer_trajectory(self, layer: int = -1) -> np.ndarray:
        hs = self.hidden_states
        return np.array([]) if len(hs) == 0 else hs[:, layer, :]

    @property
    def generation_metrics(self) -> Optional[Dict[str, float]]:
        if self._meta_cache is None:
            _ = self.meta
        metrics: Dict[str, float] = {}
        for key in ("perplexity", "entropy", "max_probability"):
            if key in self._meta_cache:
                value = self._meta_cache[key]
                if value is not None and not pd.isna(value):
                    metrics[key] = float(value)
        return metrics or None

    def get_hidden_states(
        self,
        layers: Optional[List[int]] = None,
        token_range: Optional[TokenSpan] = None,
    ) -> np.ndarray:
        hs = self.hidden_states
        if len(hs) == 0:
            return hs
        if token_range is not None:
            hs = hs[token_range.start : token_range.end]
        if layers is not None:
            hs = hs[:, layers, :]
        return hs

    def get_span_hidden_states(
        self,
        span: TokenSpan,
        layers: Optional[List[int]] = None,
        reduction: str = "none",
    ) -> np.ndarray:
        hs = self.get_hidden_states(layers=layers, token_range=span)
        if len(hs) == 0:
            return hs
        if reduction == "none":
            return hs
        if reduction == "mean":
            return hs.mean(axis=0)
        if reduction == "first":
            return hs[0]
        if reduction == "last":
            return hs[-1]
        if reduction == "sum":
            return hs.sum(axis=0)
        raise ValueError(f"Unknown reduction: {reduction}")

    def get_text_hidden_states(
        self,
        substring: str,
        layers: Optional[List[int]] = None,
        reduction: str = "mean",
        occurrence: int = -1,
    ) -> Optional[np.ndarray]:
        span = self.find_text(substring, occurrence)
        if span is None:
            return None
        return self.get_span_hidden_states(span, layers=layers, reduction=reduction)

    def has_attention(self, layer: int) -> bool:
        z = self._run._zarr
        if f"response/attn/layer_{layer}" in z:
            return True
        if "attention" in z:
            layers = list(z["attention"].attrs.get("layers", []))
            return layer in layers
        return False

    def get_attention_pattern(self, layer: int) -> Optional[np.ndarray]:
        z = self._run._zarr
        start, end = self._ptr

        if "attention" in z and "pattern" in z["attention"]:
            grp = z["attention"]
            layers = list(grp.attrs.get("layers", []))
            if layer in layers:
                li = layers.index(layer)
                return np.asarray(grp["pattern"][start:end, li])

        path = f"response/attn/layer_{layer}/pattern"
        if path not in z:
            return None
        return np.asarray(z[path][start:end])

    def get_attention_output(self, layer: int) -> Optional[np.ndarray]:
        z = self._run._zarr
        start, end = self._ptr

        if "attention" in z and "output" in z["attention"]:
            grp = z["attention"]
            layers = list(grp.attrs.get("layers", []))
            if layer in layers:
                li = layers.index(layer)
                return np.asarray(grp["output"][start:end, li])

        path = f"response/attn/layer_{layer}/output"
        if path not in z:
            return None
        return np.asarray(z[path][start:end])

    def has_mlp(self, layer: int) -> bool:
        z = self._run._zarr
        if f"response/mlp/layer_{layer}" in z:
            return True
        if "mlp" in z:
            layers = list(z["mlp"].attrs.get("layers", []))
            return layer in layers
        return False

    def get_mlp_output(self, layer: int) -> Optional[np.ndarray]:
        z = self._run._zarr
        start, end = self._ptr

        if "mlp" in z and "output" in z["mlp"]:
            grp = z["mlp"]
            layers = list(grp.attrs.get("layers", []))
            if layer in layers:
                li = layers.index(layer)
                return np.asarray(grp["output"][start:end, li])

        path = f"response/mlp/layer_{layer}/output"
        if path not in z:
            return None
        return np.asarray(z[path][start:end])

    def get_mlp_gate(self, layer: int) -> Optional[np.ndarray]:
        path = f"response/mlp/layer_{layer}/gate"
        if path not in self._run._zarr:
            return None
        start, end = self._ptr
        return np.asarray(self._run._zarr[path][start:end])

    @property
    def meta(self) -> Dict[str, Any]:
        if self._meta_cache is None:
            self._meta_cache = self._run._df.iloc[self._idx].to_dict()
        return self._meta_cache

    def get_label(self, label_name: str) -> Any:
        return self._run.get_label(self._idx, label_name)

    def summary(self) -> Dict[str, Any]:
        hs = self.hidden_states
        return {
            "index": self._idx,
            "sample_id": self.sample_id,
            "status": self.status.name,
            "n_tokens": self.n_tokens,
            "response_length": len(self.response_text),
            "finish_reason": self.finish_reason,
            "ground_truth": self.ground_truth,
            "hidden_states_shape": hs.shape if len(hs) > 0 else None,
        }

    def __repr__(self) -> str:
        return f"Sample(idx={self._idx}, status={self.status.name}, n_tokens={self.n_tokens})"
