from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HiddenStateData:
    per_token_states: Optional[np.ndarray] = None
    mean_states: Optional[np.ndarray] = None
    prompt_last_states: Optional[np.ndarray] = None
    n_tokens: int = 0

    def clear_large_arrays(self) -> None:
        self.per_token_states = None


@dataclass
class GenerationMetrics:
    max_probability: float = 0.0
    perplexity: float = float('inf')
    entropy: float = 0.0