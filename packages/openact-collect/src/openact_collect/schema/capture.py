from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaptureSpec:
    hidden_states: bool = True
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
        requested = raw or list(range(n_layers))
        resolved: List[int] = []
        for idx in requested:
            actual = n_layers + idx if idx < 0 else idx
            if 0 <= actual < n_layers:
                resolved.append(actual)
        return resolved
