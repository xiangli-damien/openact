"""Generation configuration specs for LLM inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class GenerationSpec:
    """Generation parameters for model.generate().

    When temperature > 0, do_sample is set to True in __post_init__ so that
    sampling is enabled consistently. If you set do_sample=False and
    temperature > 0, do_sample will be overridden to True (e.g. safety
    profiles with temperature=0.7 will always sample).
    """
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    do_sample: bool = False
    seed: int = 42
    stop_sequences: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.temperature > 0:
            self.do_sample = True

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GenerationSpec':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GenerationProfile(GenerationSpec):
    name: str = 'default'
    n_gen: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.n_gen < 1:
            raise ValueError(f'n_gen must be >= 1, got {self.n_gen}')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GenerationProfile':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})