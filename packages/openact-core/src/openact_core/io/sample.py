"""
Per-sample accessor for an OpenAct run.

A :class:`Sample` wraps a single row of a :class:`Run` and provides lazy
access to token IDs, character-level alignment, hidden states, and optional
attention / MLP activations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from openact_core.schema.status import SampleStatus
from openact_core.alignment.aligner import TokenAligner, TokenSpan, CharSpan

if TYPE_CHECKING:
    from openact_core.io.run import Run


class Sample:
    """Lazy accessor for a single sample inside a :class:`Run`.

    Heavy data (hidden states, token offsets) is loaded on first access and
    cached for the lifetime of the object.

    Parameters
    ----------
    run : Run
        The parent run that owns this sample.
    idx : int
        Zero-based position in the run's metadata DataFrame.
    """

    def __init__(self, run: "Run", idx: int):
        self._run = run
        self._idx = idx
        self._aligner_cache: Optional[TokenAligner] = None
        self._meta_cache: Optional[Dict[str, Any]] = None
        self._hidden_states_cache: Optional[np.ndarray] = None
        self._ptr_cache: Optional[Tuple[int, int]] = None

    # -- Identity ------------------------------------------------------------

    @property
    def index(self) -> int:
        """Zero-based index inside the parent run."""
        return self._idx

    @property
    def sample_idx(self) -> int:
        """Alias for :attr:`index` (matches parquet column name)."""
        return self._idx

    @property
    def sample_id(self) -> str:
        """String identifier stored in the parquet metadata."""
        return str(self.meta.get("sample_id", self._idx))

    @property
    def status(self) -> SampleStatus:
        status_val = self._run._df.iloc[self._idx]["status"]
        return SampleStatus(int(status_val))

    @property
    def is_valid(self) -> bool:
        return self.status == SampleStatus.OK

    # -- Text ----------------------------------------------------------------

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
        gt = self.meta.get("ground_truth")
        return str(gt) if gt is not None else None

    @property
    def finish_reason(self) -> str:
        return str(self.meta.get("finish_reason", "unknown"))

    # -- Token-level data ----------------------------------------------------

    @property
    def _ptr(self) -> Tuple[int, int]:
        """Flat start/end offsets into the zarr token arrays."""
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
        """Character ↔ token alignment for the response text."""
        if self._aligner_cache is None:
            self._aligner_cache = TokenAligner(
                text=self.response_text, offsets=self.token_offsets
            )
        return self._aligner_cache

    def token_to_text(self, token_idx: int) -> str:
        return self.aligner.token_to_text(token_idx)

    def span_to_text(self, span: TokenSpan) -> str:
        return self.aligner.token_span_to_text(span)

    def find_text(
        self, substring: str, occurrence: int = -1
    ) -> Optional[TokenSpan]:
        """Find *substring* in the response and return its token span."""
        return self.aligner.find_substring(substring, occurrence)

    def find_all_text(self, substring: str) -> List[TokenSpan]:
        return self.aligner.find_all_substrings(substring)

    # -- Hidden states -------------------------------------------------------

    @property
    def hidden_states(self) -> np.ndarray:
        """Per-token hidden states with shape ``(n_tokens, n_layers, hidden_dim)``.

        Falls back to mean states (expanded to 1-token) if per-token data is
        unavailable, or returns an empty array when nothing is stored.
        """
        if self._hidden_states_cache is None:
            start, end = self._ptr
            path = "hidden_states/per_token"
            if path in self._run._zarr:
                self._hidden_states_cache = np.asarray(
                    self._run._zarr[path][start:end]
                )
            else:
                legacy_path = "response/hidden_states/data"
                if legacy_path in self._run._zarr:
                    self._hidden_states_cache = np.asarray(
                        self._run._zarr[legacy_path][start:end]
                    )
                elif "hidden_states/mean" in self._run._zarr:
                    import warnings

                    warnings.warn(
                        f"Sample {self._idx}: per_token hidden states not "
                        f"available, returning mean states instead. Use "
                        f"sample.mean_hidden_states for explicit access.",
                        UserWarning,
                    )
                    mean_hs = np.asarray(
                        self._run._zarr["hidden_states/mean"][self._idx]
                    )
                    self._hidden_states_cache = mean_hs[np.newaxis, :, :]
                else:
                    import warnings

                    warnings.warn(
                        f"Sample {self._idx}: No hidden states available "
                        f"(per_token or mean). Returning empty array.",
                        UserWarning,
                    )
                    self._hidden_states_cache = np.array([])
        return self._hidden_states_cache

    @property
    def mean_hidden_states(self) -> np.ndarray:
        """Mean hidden states across tokens, shape ``(n_layers, hidden_dim)``."""
        if "hidden_states/mean" in self._run._zarr:
            return np.asarray(
                self._run._zarr["hidden_states/mean"][self._idx]
            )
        hs = self.hidden_states
        if len(hs) == 0:
            return np.array([])
        return hs.mean(axis=0).astype(np.float32)

    @property
    def prompt_last_hidden_states(self) -> np.ndarray:
        """Hidden states of the last prompt token (before generation)."""
        if "hidden_states/prompt_last" in self._run._zarr:
            return np.asarray(
                self._run._zarr["hidden_states/prompt_last"][self._idx]
            )
        hs = self.hidden_states
        if len(hs) == 0:
            return np.array([])
        return hs[0].astype(np.float32)

    @property
    def generated_only_hidden_states(self) -> np.ndarray:
        """Hidden states for generated tokens only.

        New format: per_token holds only generated tokens; prompt_last is stored
        in hidden_states/prompt_last. Return hidden_states as-is.

        Legacy format: per_token may include prompt_last at index 0. When
        hidden_states/prompt_last is absent, return hs[1:] so that the first
        (prompt) position is excluded and only generated tokens are returned.
        """
        hs = self.hidden_states
        if len(hs) == 0:
            return np.array([])
        # New format: prompt_last stored separately; per_token is generated-only.
        if "hidden_states/prompt_last" in self._run._zarr:
            return hs
        # Legacy: per_token was [prompt_last, gen_0, gen_1, ...]; drop index 0.
        return hs[1:] if len(hs) > 1 else np.array([])

    def get_layer_trajectory(self, layer: int = -1) -> np.ndarray:
        """Return the trajectory of a single layer across all tokens.

        Shape: ``(n_tokens, hidden_dim)``.
        """
        hs = self.hidden_states
        if len(hs) == 0:
            return np.array([])
        return hs[:, layer, :]

    @property
    def generation_metrics(self) -> Optional[Dict[str, float]]:
        if self._meta_cache is None:
            _ = self.meta
        metrics = {}
        for key in ["perplexity", "entropy", "max_probability"]:
            if key in self._meta_cache:
                val = self._meta_cache[key]
                if val is not None and not pd.isna(val):
                    metrics[key] = float(val)
        return metrics if metrics else None

    def get_hidden_states(
        self,
        layers: Optional[List[int]] = None,
        token_range: Optional[TokenSpan] = None,
    ) -> np.ndarray:
        """Slice hidden states by layers and/or token range."""
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
        """Get hidden states for a token span with optional reduction."""
        hs = self.get_hidden_states(layers=layers, token_range=span)
        if len(hs) == 0:
            return hs
        if reduction == "none":
            return hs
        elif reduction == "mean":
            return hs.mean(axis=0)
        elif reduction == "first":
            return hs[0]
        elif reduction == "last":
            return hs[-1]
        elif reduction == "sum":
            return hs.sum(axis=0)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def get_text_hidden_states(
        self,
        substring: str,
        layers: Optional[List[int]] = None,
        reduction: str = "mean",
        occurrence: int = -1,
    ) -> Optional[np.ndarray]:
        """Find *substring* in the response and return its hidden states."""
        span = self.find_text(substring, occurrence)
        if span is None:
            return None
        return self.get_span_hidden_states(
            span, layers=layers, reduction=reduction
        )

    # -- Attention / MLP (optional) ------------------------------------------

    def has_attention(self, layer: int) -> bool:
        path = f"response/attn/layer_{layer}"
        return path in self._run._zarr

    def get_attention_pattern(self, layer: int) -> Optional[np.ndarray]:
        path = f"response/attn/layer_{layer}/pattern"
        if path not in self._run._zarr:
            return None
        start, end = self._ptr
        return np.asarray(self._run._zarr[path][start:end])

    def get_attention_output(self, layer: int) -> Optional[np.ndarray]:
        path = f"response/attn/layer_{layer}/output"
        if path not in self._run._zarr:
            return None
        start, end = self._ptr
        return np.asarray(self._run._zarr[path][start:end])

    def has_mlp(self, layer: int) -> bool:
        path = f"response/mlp/layer_{layer}"
        return path in self._run._zarr

    def get_mlp_output(self, layer: int) -> Optional[np.ndarray]:
        path = f"response/mlp/layer_{layer}/output"
        if path not in self._run._zarr:
            return None
        start, end = self._ptr
        return np.asarray(self._run._zarr[path][start:end])

    def get_mlp_gate(self, layer: int) -> Optional[np.ndarray]:
        path = f"response/mlp/layer_{layer}/gate"
        if path not in self._run._zarr:
            return None
        start, end = self._ptr
        return np.asarray(self._run._zarr[path][start:end])

    # -- Metadata / labels ---------------------------------------------------

    @property
    def meta(self) -> Dict[str, Any]:
        """Full parquet row as a dict (cached on first access)."""
        if self._meta_cache is None:
            row = self._run._df.iloc[self._idx]
            self._meta_cache = row.to_dict()
        return self._meta_cache

    def get_label(self, label_name: str) -> Any:
        """Look up a label value by name (checks parquet + sidecar files)."""
        return self._run.get_label(self._idx, label_name)

    # -- Repr / summary ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Sample(idx={self._idx}, status={self.status.name}, "
            f"n_tokens={self.n_tokens})"
        )

    def summary(self) -> Dict[str, Any]:
        """Return a quick-look dict of this sample's key attributes."""
        return {
            "index": self._idx,
            "sample_id": self.sample_id,
            "status": self.status.name,
            "n_tokens": self.n_tokens,
            "response_length": len(self.response_text),
            "finish_reason": self.finish_reason,
            "ground_truth": self.ground_truth,
            "hidden_states_shape": (
                self.hidden_states.shape if len(self.hidden_states) > 0 else None
            ),
        }