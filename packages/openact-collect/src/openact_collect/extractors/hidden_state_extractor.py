from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from openact_collect.extractors.base import Extractor
from openact_collect.extractors.hidden_state_data import HiddenStateData
from openact_collect.schema import CaptureSpec


class HiddenStateExtractor(Extractor):
    def __init__(self, capture_spec: CaptureSpec, model_manager: Optional[Any] = None):
        super().__init__(capture_spec)
        self.model_manager = model_manager
        self._layer_indices: Optional[List[int]] = None
        self._attn_layer_indices: Optional[List[int]] = None
        self._mlp_layer_indices: Optional[List[int]] = None
        self._n_layers: Optional[int] = None
        self._decoder_n_layers: Optional[int] = None
        self._hidden_dim: Optional[int] = None
        self._initialized = False
        self._hf_layer_offset: Optional[int] = None

    def _lazy_init(self) -> None:
        if self._initialized or self.model_manager is None:
            return
        self._n_layers = int(self.model_manager.probed_n_layers)
        self._decoder_n_layers = int(self.model_manager.decoder_n_layers)
        self._hidden_dim = int(self.model_manager.get_model_spec().hidden_dim or 0)
        self._layer_indices = self.capture_spec.get_effective_layers(self._n_layers)
        self._attn_layer_indices = self.capture_spec.get_effective_attention_layers(self._decoder_n_layers)
        self._mlp_layer_indices = self.capture_spec.get_effective_mlp_layers(self._decoder_n_layers)
        self._initialized = True

    @property
    def layer_indices(self) -> List[int]:
        self._lazy_init()
        return self._layer_indices or []

    def setup(self, model: Any, model_manager: Any) -> None:
        self.model_manager = model_manager
        self._initialized = False
        self._lazy_init()

    def cleanup(self) -> None:
        self._remove_hooks()
        self._model = None

    def _infer_hf_layer_offset(self, step_hs: Tuple[torch.Tensor, ...]) -> int:
        if self._n_layers is None:
            return 0
        hidden_state_len = len(step_hs)
        if hidden_state_len == self._n_layers + 1:
            return 1
        if hidden_state_len >= self._n_layers:
            return hidden_state_len - self._n_layers
        return 0

    def _get_layer_vector(self, step_hs: Tuple[torch.Tensor, ...], layer_idx: int, target_dtype: torch.dtype) -> torch.Tensor:
        if self._hf_layer_offset is None:
            self._hf_layer_offset = self._infer_hf_layer_offset(step_hs)
        actual_idx = layer_idx + self._hf_layer_offset
        if actual_idx < 0 or actual_idx >= len(step_hs):
            return torch.zeros(self._hidden_dim, dtype=target_dtype, device=step_hs[0].device)
        tensor = step_hs[actual_idx]
        if tensor.dim() == 3:
            return tensor[0, -1, :].to(target_dtype)
        if tensor.dim() == 2:
            return tensor[-1, :].to(target_dtype)
        if tensor.dim() == 1:
            return tensor.to(target_dtype)
        return tensor.flatten()[: self._hidden_dim].to(target_dtype)

    def _extract_step_matrix_torch(self, step_hs: Tuple[torch.Tensor, ...], target_dtype: torch.dtype) -> torch.Tensor:
        vectors = [self._get_layer_vector(step_hs, layer_idx, target_dtype) for layer_idx in self.layer_indices]
        if not vectors:
            return torch.zeros((0, self._hidden_dim), dtype=target_dtype)
        return torch.stack(vectors, dim=0)

    def _extract_prompt_last_via_forward(self, input_ids: torch.Tensor) -> np.ndarray:
        self._lazy_init()
        n_layers = len(self.layer_indices)
        hidden_dim = int(self._hidden_dim or 0)
        out = np.zeros((n_layers, hidden_dim), dtype=np.float32)
        if self.model_manager is None:
            return out
        with torch.inference_mode():
            outputs = self.model_manager.model(input_ids=input_ids.to(self.model_manager.device), output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states or ()
        if not hidden_states:
            return out
        hf_offset = self._infer_hf_layer_offset(hidden_states)
        for index, layer_idx in enumerate(self.layer_indices):
            actual_idx = layer_idx + hf_offset
            if 0 <= actual_idx < len(hidden_states):
                tensor = hidden_states[actual_idx]
                if tensor.dim() == 3:
                    vector = tensor[0, -1, :]
                elif tensor.dim() == 2:
                    vector = tensor[-1, :]
                else:
                    vector = tensor.flatten()[:hidden_dim]
                out[index] = vector.to(torch.float32).cpu().numpy()
        return out

    def extract(self, gen_result: Any, input_ids: Optional[torch.Tensor] = None) -> HiddenStateData:
        self._lazy_init()
        hidden_states = gen_result.hidden_states or []
        keep_indices = list(gen_result.keep_indices or [])
        traces = getattr(gen_result, 'traces', None)
        if not keep_indices:
            token_count = len(getattr(gen_result, 'token_ids', []) or [])
            if traces is not None and traces.n_steps() > 0:
                if traces.n_steps() == token_count + 1:
                    keep_indices = list(range(1, traces.n_steps()))
                else:
                    keep_indices = list(range(min(traces.n_steps(), token_count)))
            else:
                keep_indices = list(range(token_count))
        per_token: Optional[np.ndarray] = None
        mean_states: Optional[np.ndarray] = None
        prompt_last: Optional[np.ndarray] = None
        n_tokens = 0
        if self.capture_spec.hidden_states and hidden_states and keep_indices:
            self._hf_layer_offset = None
            matrices_f32 = [self._extract_step_matrix_torch(hidden_states[idx], torch.float32) for idx in keep_indices if idx < len(hidden_states)]
            if matrices_f32:
                n_tokens = len(matrices_f32)
                if self.capture_spec.save_mean_states:
                    mean_states = torch.stack(matrices_f32, dim=0).mean(dim=0).cpu().numpy().astype(np.float32)
                if self.capture_spec.save_per_token:
                    out_dtype = np.float16 if self.capture_spec.hidden_states_dtype == 'float16' else np.float32
                    per_token = torch.stack(matrices_f32, dim=0).cpu().numpy().astype(out_dtype)
                if self.capture_spec.save_prompt_last:
                    has_prompt_step = len(hidden_states) == gen_result.generated_length + 1
                    if has_prompt_step:
                        prompt_last = self._extract_step_matrix_torch(hidden_states[0], torch.float32).cpu().numpy().astype(np.float32)
                    elif input_ids is not None:
                        prompt_last = self._extract_prompt_last_via_forward(input_ids)
        elif self.capture_spec.save_prompt_last and input_ids is not None and self.capture_spec.hidden_states:
            prompt_last = self._extract_prompt_last_via_forward(input_ids)
        if n_tokens == 0:
            n_tokens = len(getattr(gen_result, 'token_ids', []) or [])
        attn_out = self._extract_trace_outputs(traces.attn_outputs if traces is not None else None, keep_indices, self._attn_layer_indices or [], int(self._hidden_dim or 0))
        mlp_out = self._extract_trace_outputs(traces.mlp_outputs if traces is not None else None, keep_indices, self._mlp_layer_indices or [], int(self._hidden_dim or 0))
        attn_pat = self._extract_trace_patterns(traces.attn_patterns if traces is not None else None, keep_indices, self._attn_layer_indices or [], int(getattr(self.capture_spec, 'attention_pattern_window', 256) or 256))
        return HiddenStateData(per_token_states=per_token, mean_states=mean_states, prompt_last_states=prompt_last, attention_outputs=attn_out, attention_patterns=attn_pat, mlp_outputs=mlp_out, n_tokens=n_tokens)

    @staticmethod
    def _extract_trace_outputs(trace_steps: Optional[List[Dict[int, torch.Tensor]]], keep_indices: List[int], layer_indices: List[int], hidden_dim: int) -> Optional[np.ndarray]:
        if not trace_steps or not keep_indices or not layer_indices:
            return None
        n_tokens = len(keep_indices)
        n_layers = len(layer_indices)
        out = np.zeros((n_tokens, n_layers, int(hidden_dim)), dtype=np.float16)
        for token_index, step_idx in enumerate(keep_indices):
            if step_idx >= len(trace_steps):
                continue
            step = trace_steps[step_idx]
            for layer_pos, layer in enumerate(layer_indices):
                vector = step.get(layer)
                if vector is None:
                    continue
                values = vector.detach().cpu().numpy().astype(np.float16, copy=False)
                if values.shape[0] >= hidden_dim:
                    out[token_index, layer_pos, :] = values[:hidden_dim]
                else:
                    out[token_index, layer_pos, : values.shape[0]] = values
        return out

    @staticmethod
    def _extract_trace_patterns(trace_steps: Optional[List[Dict[int, torch.Tensor]]], keep_indices: List[int], layer_indices: List[int], window: int) -> Optional[np.ndarray]:
        if not trace_steps or not keep_indices or not layer_indices:
            return None
        n_heads: Optional[int] = None
        for step in trace_steps:
            for value in step.values():
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    n_heads = int(value.shape[0])
                    break
            if n_heads is not None:
                break
        if n_heads is None:
            return None
        n_tokens = len(keep_indices)
        n_layers = len(layer_indices)
        out = np.zeros((n_tokens, n_layers, n_heads, int(window)), dtype=np.float16)
        for token_index, step_idx in enumerate(keep_indices):
            if step_idx >= len(trace_steps):
                continue
            step = trace_steps[step_idx]
            for layer_pos, layer in enumerate(layer_indices):
                pattern = step.get(layer)
                if pattern is None:
                    continue
                values = pattern.detach().cpu().numpy().astype(np.float16, copy=False)
                if values.shape[0] != n_heads:
                    continue
                out[token_index, layer_pos, :, :] = values[:, -window:]
        return out
