from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from openact_collect.schema import CaptureSpec
from openact_collect.tracing.module_resolver import find_transformer_layers, resolve_attention_module, resolve_mlp_module

logger = logging.getLogger(__name__)


@dataclass
class ActivationTrace:
    attn_outputs: Optional[List[Dict[int, torch.Tensor]]] = None
    attn_patterns: Optional[List[Dict[int, torch.Tensor]]] = None
    mlp_outputs: Optional[List[Dict[int, torch.Tensor]]] = None

    def n_steps(self) -> int:
        for sequence in (self.attn_outputs, self.attn_patterns, self.mlp_outputs):
            if sequence is not None:
                return len(sequence)
        return 0


class ActivationRecorder:
    def __init__(self, model: Any, capture_spec: CaptureSpec, decoder_n_layers: Optional[int] = None):
        self.model = model
        self.capture_spec = capture_spec
        self.decoder_n_layers = decoder_n_layers
        self._hooks: List[Any] = []
        self._step_idx = -1
        self._layers: Optional[List[Any]] = None
        self._attn_layers: Optional[List[int]] = capture_spec.attention_layers
        self._mlp_layers: Optional[List[int]] = capture_spec.mlp_layers
        self._want_attn_out = bool(capture_spec.attention and capture_spec.attention_save_outputs)
        self._want_attn_pat = bool(capture_spec.attention and capture_spec.attention_save_patterns)
        self._want_mlp_out = bool(capture_spec.mlp and capture_spec.mlp_save_output)
        self._attn_window = int(getattr(capture_spec, 'attention_pattern_window', 256) or 256)
        self._attn_outputs_steps: Optional[List[Dict[int, torch.Tensor]]] = [] if self._want_attn_out else None
        self._attn_patterns_steps: Optional[List[Dict[int, torch.Tensor]]] = [] if self._want_attn_pat else None
        self._mlp_outputs_steps: Optional[List[Dict[int, torch.Tensor]]] = [] if self._want_mlp_out else None

    def __enter__(self) -> 'ActivationRecorder':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def start(self) -> None:
        if not (self._want_attn_out or self._want_attn_pat or self._want_mlp_out):
            return
        self._layers = find_transformer_layers(self.model)
        decoder_n_layers = int(self.decoder_n_layers or len(self._layers))
        if self.capture_spec.attention_layers is not None:
            self._attn_layers = self.capture_spec.get_effective_attention_layers(decoder_n_layers)
        if self.capture_spec.mlp_layers is not None:
            self._mlp_layers = self.capture_spec.get_effective_mlp_layers(decoder_n_layers)
        self._hooks.append(self.model.register_forward_pre_hook(self._on_model_pre_forward))
        for layer_idx, layer in enumerate(self._layers):
            attn_enabled = self._want_attn_out or self._want_attn_pat
            if self._attn_layers is not None and layer_idx not in self._attn_layers:
                attn_enabled = False
            mlp_enabled = self._want_mlp_out
            if self._mlp_layers is not None and layer_idx not in self._mlp_layers:
                mlp_enabled = False
            if attn_enabled:
                attention_module = resolve_attention_module(layer)
                if attention_module is not None:
                    self._hooks.append(attention_module.register_forward_hook(self._make_attn_hook(layer_idx)))
            if mlp_enabled:
                mlp_module = resolve_mlp_module(layer)
                if mlp_module is not None:
                    self._hooks.append(mlp_module.register_forward_hook(self._make_mlp_hook(layer_idx)))

    def stop(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()

    def trace(self) -> Optional[ActivationTrace]:
        if not (self._want_attn_out or self._want_attn_pat or self._want_mlp_out):
            return None
        return ActivationTrace(attn_outputs=self._attn_outputs_steps, attn_patterns=self._attn_patterns_steps, mlp_outputs=self._mlp_outputs_steps)

    def _on_model_pre_forward(self, _module: Any, _inputs: Any) -> None:
        self._step_idx += 1
        if self._attn_outputs_steps is not None:
            self._attn_outputs_steps.append({})
        if self._attn_patterns_steps is not None:
            self._attn_patterns_steps.append({})
        if self._mlp_outputs_steps is not None:
            self._mlp_outputs_steps.append({})

    @staticmethod
    def _as_tensor(output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first
        return None

    @staticmethod
    def _looks_like_attention_weights(tensor: Any) -> bool:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            return False
        if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[-1] < 1:
            return False
        try:
            probe = tensor[0, 0, -1].detach().float()
        except Exception:
            return False
        if probe.numel() == 0 or not torch.isfinite(probe).all():
            return False
        total = float(probe.sum().item())
        return 0.5 <= total <= 1.5

    @classmethod
    def _as_attn_weights(cls, output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            weights = output[1]
            if cls._looks_like_attention_weights(weights):
                return weights
        return None

    def _make_attn_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            step_idx = self._step_idx
            if step_idx < 0:
                return
            if self._attn_outputs_steps is not None:
                tensor = self._as_tensor(output)
                if tensor is not None and tensor.ndim >= 3:
                    self._attn_outputs_steps[step_idx][layer_idx] = tensor[0, -1].detach().to('cpu', dtype=torch.float16)
            if self._attn_patterns_steps is None:
                return
            weights = self._as_attn_weights(output)
            if weights is None:
                return
            if weights.ndim == 4:
                weights_last = weights[0, :, -1, :]
            elif weights.ndim == 3:
                weights_last = weights[0, :, :]
            else:
                return
            key_len = weights_last.shape[-1]
            window = self._attn_window
            if key_len >= window:
                weights_window = weights_last[..., -window:]
            else:
                pad = window - key_len
                weights_window = torch.nn.functional.pad(weights_last, (pad, 0), value=0.0)
            self._attn_patterns_steps[step_idx][layer_idx] = weights_window.detach().to('cpu', dtype=torch.float16)
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            step_idx = self._step_idx
            if step_idx < 0 or self._mlp_outputs_steps is None:
                return
            tensor = self._as_tensor(output)
            if tensor is None or tensor.ndim < 3:
                return
            self._mlp_outputs_steps[step_idx][layer_idx] = tensor[0, -1].detach().to('cpu', dtype=torch.float16)
        return hook
