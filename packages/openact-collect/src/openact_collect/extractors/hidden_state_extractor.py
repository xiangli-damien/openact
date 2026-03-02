from typing import Any, List, Optional, Set, Tuple

import warnings
import numpy as np
import torch

from openact_collect.schema import CaptureSpec
from openact_collect.extractors.base import Extractor
from openact_collect.extractors.hidden_state_data import HiddenStateData


class HiddenStateExtractor(Extractor):

    def __init__(self, capture_spec: CaptureSpec, model_manager: Optional[Any] = None):
        super().__init__(capture_spec)
        self.model_manager = model_manager
        self._layer_indices: Optional[List[int]] = None
        self._n_layers: Optional[int] = None
        self._hidden_dim: Optional[int] = None
        self._initialized = False
        self._warned_layer_oob: Set[int] = set()
        self._hf_layer_offset: Optional[int] = None

    def _lazy_init(self) -> None:
        if self._initialized or self.model_manager is None:
            return
        self._n_layers = self.model_manager.probed_n_layers
        model_spec = self.model_manager.get_model_spec()
        self._hidden_dim = model_spec.hidden_dim
        raw_indices = self.get_layer_indices(self._n_layers)
        self._layer_indices = self._normalize_indices(raw_indices, self._n_layers)
        self._initialized = True

    def _get_output_dtypes(self) -> Tuple[type, torch.dtype]:
        dtype = (
            np.float16
            if self.capture_spec.hidden_states_dtype == "float16"
            else np.float32
        )
        torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
        return dtype, torch_dtype

    def _get_extract_dims(self, keep_indices: List[int]) -> Tuple[int, int, int]:
        return len(keep_indices), len(self._layer_indices), self._hidden_dim

    @staticmethod
    def _normalize_indices(indices: List[int], n_layers: int) -> List[int]:
        normalized = []
        for idx in indices:
            resolved = n_layers + idx if idx < 0 else idx
            if 0 <= resolved < n_layers:
                normalized.append(resolved)
        return normalized

    def _infer_hf_layer_offset(self, step_hs: Tuple[torch.Tensor, ...]) -> int:
        if self._n_layers is None:
            return 0
        hs_len = len(step_hs)
        if hs_len == self._n_layers + 1:
            return 1
        if hs_len == self._n_layers:
            return 0
        if hs_len > self._n_layers:
            return max(0, hs_len - self._n_layers)
        return 0

    def _get_layer_vector(
        self, step_hs: Tuple[torch.Tensor, ...], layer_idx: int
    ) -> Optional[torch.Tensor]:
        if self._hf_layer_offset is None:
            self._hf_layer_offset = self._infer_hf_layer_offset(step_hs)
        actual_idx = layer_idx + self._hf_layer_offset
        if actual_idx < 0 or actual_idx >= len(step_hs):
            if layer_idx not in self._warned_layer_oob:
                warnings.warn(
                    f"Layer index {layer_idx} (HF actual {actual_idx}) out of range "
                    f"for step hidden states (len={len(step_hs)}). "
                    f"Check capture_spec layer indices vs model depth.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_layer_oob.add(layer_idx)
            return None
        t = step_hs[actual_idx]
        if t is None:
            return None
        if t.dim() == 3:
            return t[0, -1, :]
        if t.dim() == 2:
            return t[-1, :]
        if t.dim() == 1:
            return t
        return None

    def _extract_step_matrix_torch(
        self, step_hs: Tuple[torch.Tensor, ...], target_dtype: torch.dtype
    ) -> torch.Tensor:
        vecs: List[torch.Tensor] = []
        device = step_hs[0].device if len(step_hs) > 0 else torch.device("cpu")
        for li in self._layer_indices:
            v = self._get_layer_vector(step_hs, li)
            if v is None:
                vecs.append(
                    torch.zeros(self._hidden_dim, device=device, dtype=target_dtype)
                )
            else:
                vecs.append(v.to(target_dtype) if v.dtype != target_dtype else v)
        return torch.stack(vecs, dim=0)

    def _extract_step_matrix(
        self, step_hs: Tuple[torch.Tensor, ...], target_dtype: torch.dtype
    ) -> np.ndarray:
        return self._extract_step_matrix_torch(step_hs, target_dtype).cpu().numpy()

    def setup(self, model: Any, model_manager: Any) -> None:
        warnings.warn(
            "HiddenStateExtractor.setup() is deprecated. "
            "Pass model_manager to __init__ instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._model = model
        self.model_manager = model_manager
        self._lazy_init()

    def extract(self, gen_result: Any) -> HiddenStateData:
        self._lazy_init()
        hidden_states = gen_result.hidden_states
        keep_indices = gen_result.keep_indices
        n_tokens = len(keep_indices) if keep_indices else 0

        if not hidden_states or n_tokens == 0:
            return HiddenStateData(n_tokens=0)

        if keep_indices:
            max_idx = max(keep_indices)
            if max_idx >= len(hidden_states):
                warnings.warn(
                    f"Hidden state extraction failed: keep_indices max={max_idx} >= "
                    f"len(hidden_states)={len(hidden_states)}. "
                    f"Model/transformers mismatch or corrupted output.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return HiddenStateData(n_tokens=0)

        per_token = None
        mean_states = None

        if self.capture_spec.save_per_token and self.capture_spec.save_mean_states:
            per_token, mean_states = self._extract_per_token_and_mean(
                hidden_states, keep_indices
            )
        else:
            if self.capture_spec.save_per_token:
                per_token = self._extract_per_token(hidden_states, keep_indices)
            if self.capture_spec.save_mean_states:
                mean_states = self._compute_mean_states(hidden_states, keep_indices)

        return HiddenStateData(
            per_token_states=per_token,
            mean_states=mean_states,
            prompt_last_states=None,
            n_tokens=n_tokens,
        )

    def extract_prompt_last(self, input_ids: torch.Tensor) -> np.ndarray:
        self._lazy_init()
        n_layers = len(self._layer_indices)
        H = self._hidden_dim
        out = np.zeros((n_layers, H), dtype=np.float32)

        if self.model_manager is None:
            return out

        device = self.model_manager.device
        with torch.inference_mode():
            outputs = self.model_manager.model(
                input_ids=input_ids.to(device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        prompt_hidden_states = outputs.hidden_states
        if not prompt_hidden_states:
            return out

        hf_offset = self._infer_hf_layer_offset(prompt_hidden_states)

        for i, li in enumerate(self._layer_indices):
            actual_idx = li + hf_offset
            if 0 <= actual_idx < len(prompt_hidden_states):
                t = prompt_hidden_states[actual_idx]
                if t.dim() == 3:
                    vec = t[0, -1, :].to(torch.float32).cpu().numpy()
                elif t.dim() == 2:
                    vec = t[-1, :].to(torch.float32).cpu().numpy()
                elif t.dim() == 1:
                    vec = t.to(torch.float32).cpu().numpy()
                else:
                    vec = t.flatten()[: self._hidden_dim].to(torch.float32).cpu().numpy()
                out[i, :] = vec

        return out

    def _extract_per_token_and_mean(
        self,
        hidden_states: List[Tuple[torch.Tensor, ...]],
        keep_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_tokens, n_layers, H = self._get_extract_dims(keep_indices)
        dtype, torch_dtype = self._get_output_dtypes()

        out_indices: List[int] = []
        mats: List[torch.Tensor] = []
        for out_idx, step_idx in enumerate(keep_indices):
            if step_idx >= len(hidden_states):
                continue
            out_indices.append(out_idx)
            mats.append(
                self._extract_step_matrix_torch(hidden_states[step_idx], torch_dtype)
            )

        out = np.zeros((n_tokens, n_layers, H), dtype=dtype)
        if not mats:
            return out, np.zeros((n_layers, H), dtype=np.float32)

        stacked = torch.stack(mats, dim=0)
        arr = stacked.cpu().numpy().astype(dtype)
        for i, out_idx in enumerate(out_indices):
            out[out_idx, :, :] = arr[i]

        mean_states = stacked.float().mean(dim=0).cpu().numpy().astype(np.float32)
        return out, mean_states

    def _extract_per_token(
        self,
        hidden_states: List[Tuple[torch.Tensor, ...]],
        keep_indices: List[int],
    ) -> np.ndarray:
        n_tokens, n_layers, H = self._get_extract_dims(keep_indices)
        dtype, torch_dtype = self._get_output_dtypes()

        out_indices: List[int] = []
        mats: List[torch.Tensor] = []
        for out_idx, step_idx in enumerate(keep_indices):
            if step_idx >= len(hidden_states):
                continue
            out_indices.append(out_idx)
            mats.append(
                self._extract_step_matrix_torch(hidden_states[step_idx], torch_dtype)
            )

        out = np.zeros((n_tokens, n_layers, H), dtype=dtype)
        if not mats:
            return out
        stacked = torch.stack(mats, dim=0)
        arr = stacked.cpu().numpy().astype(dtype)
        for i, out_idx in enumerate(out_indices):
            out[out_idx, :, :] = arr[i]
        return out

    def _compute_mean_states(
        self,
        hidden_states: List[Tuple[torch.Tensor, ...]],
        keep_indices: List[int],
    ) -> np.ndarray:
        _, n_layers, H = self._get_extract_dims(keep_indices)
        mats: List[torch.Tensor] = []
        for step_idx in keep_indices:
            if step_idx >= len(hidden_states):
                continue
            mats.append(
                self._extract_step_matrix_torch(hidden_states[step_idx], torch.float32)
            )
        if not mats:
            return np.zeros((n_layers, H), dtype=np.float32)
        stacked = torch.stack(mats, dim=0)
        return stacked.mean(dim=0).cpu().numpy().astype(np.float32)

    def cleanup(self) -> None:
        self._remove_hooks()
        self._model = None

    @property
    def layer_indices(self) -> List[int]:
        self._lazy_init()
        return self._layer_indices or []