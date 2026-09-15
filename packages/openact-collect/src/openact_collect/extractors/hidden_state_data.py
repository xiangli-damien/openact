from dataclasses import dataclass
from typing import Optional
import numpy as np
@dataclass
class HiddenStateData:
    per_token_states: Optional[np.ndarray] = None
    mean_states: Optional[np.ndarray] = None
    prompt_last_states: Optional[np.ndarray] = None
    # Optional additional activations.
    attention_outputs: Optional[np.ndarray] = None
    attention_patterns: Optional[np.ndarray] = None
    mlp_outputs: Optional[np.ndarray] = None
    final_norm_pre: Optional[np.ndarray] = None
    final_norm_post: Optional[np.ndarray] = None
    final_norm_pre_mean: Optional[np.ndarray] = None
    final_norm_post_mean: Optional[np.ndarray] = None
    final_norm_pre_prompt_last: Optional[np.ndarray] = None
    final_norm_post_prompt_last: Optional[np.ndarray] = None
    n_tokens: int = 0
    def clear_large_arrays(self) -> None:
        self.per_token_states = None
        self.attention_outputs = None
        self.attention_patterns = None
        self.mlp_outputs = None
        self.final_norm_pre = None
        self.final_norm_post = None
@dataclass
class GenerationMetrics:
    max_probability: float = 0.0
    perplexity: float = float('inf')
    entropy: float = 0.0
