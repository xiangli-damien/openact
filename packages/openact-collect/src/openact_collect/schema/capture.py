from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CaptureSpec:
    hidden_states: bool = True
    hidden_states_layers: Optional[List[int]] = None
    hidden_states_dtype: str = 'float16'
    save_per_token: bool = True
    save_mean_states: bool = True
    save_prompt_last: bool = True
    compute_online_metrics: bool = True
    attention: bool = False
    attention_layers: Optional[List[int]] = None
    attention_save_patterns: bool = False
    attention_save_outputs: bool = True
    mlp: bool = False
    mlp_layers: Optional[List[int]] = None
    mlp_save_gate: bool = False
    mlp_save_output: bool = True
    custom_selectors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CaptureSpec':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_effective_layers(self, n_layers: int) -> List[int]:
        if self.hidden_states_layers is not None:
            return self.hidden_states_layers
        return list(range(n_layers))
