import math
import torch
from transformers.generation.logits_process import LogitsProcessor
class OnlineMetricsProcessor(LogitsProcessor):
    def __init__(self, greedy: bool = True):
        self.greedy = greedy
        self.n_steps = 0
        self.sum_max_prob = 0.0
        self.sum_entropy = 0.0
        self.sum_logprob = 0.0
    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        logits = scores[0].to(torch.float32)
        probs = torch.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        self.sum_max_prob += max_prob
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum().item()
        self.sum_entropy += entropy
        if self.greedy:
            chosen_token = torch.argmax(logits).item()
            chosen_prob = probs[chosen_token].clamp_min(1e-12)
            self.sum_logprob += torch.log(chosen_prob).item()
        self.n_steps += 1
        return scores
    def finalize(self) -> "GenerationMetrics":
        from openact_collect.extractors.hidden_state_data import GenerationMetrics
        if self.n_steps == 0:
            return GenerationMetrics(
                max_probability=0.0, entropy=0.0, perplexity=float("inf")
            )
        avg_max_prob = self.sum_max_prob / self.n_steps
        avg_entropy = self.sum_entropy / self.n_steps
        if self.greedy and self.n_steps > 0:
            avg_nll = -self.sum_logprob / self.n_steps
            perplexity = math.exp(min(avg_nll, 100))
        else:
            perplexity = float("inf")
        return GenerationMetrics(
            max_probability=avg_max_prob,
            entropy=avg_entropy,
            perplexity=perplexity,
        )
    def reset(self) -> None:
        self.n_steps = 0
        self.sum_max_prob = 0.0
        self.sum_entropy = 0.0
        self.sum_logprob = 0.0
