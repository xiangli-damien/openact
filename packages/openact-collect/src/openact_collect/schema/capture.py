from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaptureSpec:
    hidden_states: bool = True
    # Capture the input and output of the model's final normalization module.
    final_norm: bool = True
    hidden_states_layers: Optional[List[int]] = None
    hidden_states_dtype: str = "float16"
    save_per_token: bool = True
    save_mean_states: bool = True
    save_prompt_last: bool = True
    compute_online_metrics: bool = True
    attention: bool = False
    attention_layers: Optional[List[int]] = None
    attention_save_patterns: bool = False
    attention_save_outputs: bool = True
    # When saving attention patterns, store only the last-token attention weights
    # over the most recent K keys. This keeps storage fixed-size.
    attention_pattern_window: int = 256
    mlp: bool = False
    mlp_layers: Optional[List[int]] = None
    mlp_save_gate: bool = False
    mlp_save_output: bool = True
    custom_selectors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for selection in (self.hidden_states_layers, self.attention_layers, self.mlp_layers):
            if selection is not None and (not selection or any(type(index) is not int for index in selection)):
                raise ValueError('Layer selections must be nonempty lists of integers')
        if self.hidden_states_dtype not in ('float16', 'float32'):
            raise ValueError('hidden_states_dtype must be float16 or float32')
        if self.attention_pattern_window < 1:
            raise ValueError('attention_pattern_window must be positive')
        if self.mlp_save_gate or self.custom_selectors:
            raise ValueError('MLP gate capture and custom selectors are not implemented')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptureSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_effective_layers(self, n_layers: int) -> List[int]:
        return self._resolve_indices(self.hidden_states_layers, n_layers)

    def get_effective_attention_layers(self, n_layers: int) -> List[int]:
        return self._resolve_indices(self.attention_layers, n_layers)

    def get_effective_mlp_layers(self, n_layers: int) -> List[int]:
        return self._resolve_indices(self.mlp_layers, n_layers)

    @staticmethod
    def _resolve_indices(raw: Optional[List[int]], n_layers: int) -> List[int]:
        requested = list(range(n_layers)) if raw is None else raw
        if not requested:
            raise ValueError('Layer selection must not be empty')
        resolved: List[int] = []
        for idx in requested:
            actual = n_layers + idx if idx < 0 else idx
            if not 0 <= actual < n_layers:
                raise ValueError(f'Layer {idx} is outside the available {n_layers} layers')
            if actual in resolved:
                raise ValueError(f'Duplicate layer selection: {idx}')
            resolved.append(actual)
        return resolved
