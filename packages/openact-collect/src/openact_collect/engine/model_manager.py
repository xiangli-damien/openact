import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessorList

from openact_collect.schema import GenerationSpec
from openact_core.schema.manifest import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    input_ids: torch.Tensor
    generated_tokens: torch.Tensor
    full_sequence: torch.Tensor
    input_length: int
    generated_length: int
    hidden_states: List[Tuple[torch.Tensor, ...]]
    scores: Optional[List[torch.Tensor]]
    finish_reason: str
    token_ids: List[int]
    keep_indices: List[int]


class ModelRunner:

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.dtype_str = dtype
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation

        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._config = None
        self._terminator_ids: List[int] = []
        self._probed_n_layers: Optional[int] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        dtype = self._resolve_dtype(self.dtype_str)
        self._config = AutoConfig.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code
        )

        model_kwargs: Dict[str, Any] = {
            "config": self._config,
            "torch_dtype": dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **model_kwargs
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._terminator_ids = self._get_termination_tokens()
        self._probed_n_layers = self._probe_n_layers()
        self._loaded = True

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.float16)

    def _get_termination_tokens(self) -> List[int]:
        terminators = [self._tokenizer.eos_token_id]

        additional = getattr(self._tokenizer, "additional_special_tokens_ids", None)
        if additional:
            for tid in additional:
                if tid not in terminators:
                    terminators.append(tid)

        for token in ["<|eot_id|>", "<|im_end|>", "</s>"]:
            try:
                tid = self._tokenizer.convert_tokens_to_ids(token)
                if tid is not None and tid != self._tokenizer.unk_token_id:
                    if tid not in terminators:
                        terminators.append(tid)
            except Exception:
                logger.debug(
                    "Could not resolve termination token %r", token, exc_info=True
                )

        return terminators

    def _probe_n_layers(self) -> int:
        try:
            tok = self._tokenizer("probe", return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self._model(
                    **tok, output_hidden_states=True, use_cache=False, return_dict=True
                )
            hs = out.hidden_states
            if isinstance(hs, (list, tuple)):
                return len(hs)
        except Exception:
            logger.debug(
                "Probe forward failed, falling back to config num_hidden_layers",
                exc_info=True,
            )

        cfg_layers = (
            getattr(self._config, "num_hidden_layers", None)
            or getattr(self._config, "n_layer", None)
            or getattr(self._config, "n_layers", None)
            or 0
        )
        return cfg_layers + 1

    @property
    def model(self) -> PreTrainedModel:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if not self._loaded:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def probed_n_layers(self) -> int:
        if self._probed_n_layers is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._probed_n_layers

    def get_model_spec(self) -> ModelSpec:
        cfg = self._config
        hidden_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "d_model", None)
        )
        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        return ModelSpec(
            name=Path(self.model_name_or_path).name,
            source="huggingface" if "/" in self.model_name_or_path else "local",
            identifier=self.model_name_or_path,
            dtype=self.dtype_str,
            architecture=(
                cfg.architectures[0]
                if hasattr(cfg, "architectures") and cfg.architectures
                else None
            ),
            n_layers=self._probed_n_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            vocab_size=cfg.vocab_size if hasattr(cfg, "vocab_size") else None,
            tokenizer_class=type(self._tokenizer).__name__,
        )

    def apply_chat_template(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> torch.Tensor:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
        text = "\n".join(m["content"] for m in messages)
        return self.tokenizer(text, return_tensors="pt").input_ids

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _align_hidden_states(
        self,
        hidden_states: List[Tuple[torch.Tensor, ...]],
        generated_length: int,
    ) -> List[Tuple[torch.Tensor, ...]]:
        if not hidden_states:
            return hidden_states
        hs_len = len(hidden_states)
        if hs_len == generated_length:
            return hidden_states
        if hs_len == generated_length + 1:
            return hidden_states
        if hs_len > generated_length + 1:
            warnings.warn(
                f"Hidden states length {hs_len} > generated_length+1 ({generated_length + 1}); "
                f"keeping last {generated_length + 1} steps.",
                UserWarning,
                stacklevel=2,
            )
            return hidden_states[hs_len - (generated_length + 1):]
        if hs_len > 0:
            warnings.warn(
                f"Hidden states length {hs_len} < generated_length {generated_length}; "
                f"some tokens will lack hidden states.",
                UserWarning,
                stacklevel=2,
            )
        return hidden_states

    def _build_keep_indices(
        self,
        gen_ids: List[int],
        aligned_hs_len: int,
        generated_length: int,
    ) -> Tuple[List[int], List[int]]:
        keep_indices: List[int] = []
        token_ids: List[int] = []
        has_prompt_step = aligned_hs_len == generated_length + 1

        for i, tid in enumerate(gen_ids):
            step_idx = i + 1 if has_prompt_step else i
            if step_idx < aligned_hs_len:
                keep_indices.append(step_idx)
                token_ids.append(tid)
            else:
                break

        if len(keep_indices) < len(gen_ids):
            warnings.warn(
                "Some generated tokens are missing hidden states due to alignment mismatch.",
                UserWarning,
                stacklevel=2,
            )

        return keep_indices, token_ids

    def generate(
        self,
        input_ids: torch.Tensor,
        gen_spec: GenerationSpec,
        output_scores: bool = False,
        logits_processors: Optional[List[Any]] = None,
        capture_hidden_states: bool = True,
    ) -> GenerationResult:
        device = self.device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        input_length = input_ids.shape[1]

        self._set_seed(int(gen_spec.seed))

        do_sample = bool(gen_spec.do_sample) and gen_spec.temperature > 0

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(gen_spec.max_new_tokens),
            "do_sample": do_sample,
            "eos_token_id": self._terminator_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": output_scores,
            "output_hidden_states": bool(capture_hidden_states),
        }

        if do_sample:
            gen_kwargs["temperature"] = float(gen_spec.temperature)
            gen_kwargs["top_p"] = float(gen_spec.top_p)
            if gen_spec.top_k is not None:
                gen_kwargs["top_k"] = int(gen_spec.top_k)
        else:
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = None

        if logits_processors:
            gen_kwargs["logits_processor"] = LogitsProcessorList(logits_processors)

        with torch.inference_mode():
            output = self.model.generate(**gen_kwargs)

        sequences = output.sequences
        gen_ids = sequences[0, input_length:].tolist()
        generated_length = len(gen_ids)

        generated_tokens = sequences[:, input_length:]
        full_sequence = sequences

        hidden_states = output.hidden_states if hasattr(output, "hidden_states") else []
        scores = output.scores if output_scores and hasattr(output, "scores") else None

        if capture_hidden_states and hidden_states:
            hidden_states = self._align_hidden_states(hidden_states, generated_length)
            aligned_hs_len = len(hidden_states)
            keep_indices, token_ids = self._build_keep_indices(
                gen_ids, aligned_hs_len, generated_length
            )
        else:
            hidden_states = []
            keep_indices = []
            token_ids = list(gen_ids)

        finish_reason = self._determine_finish_reason(
            generated_tokens, generated_length, gen_spec.max_new_tokens
        )

        return GenerationResult(
            input_ids=input_ids,
            generated_tokens=generated_tokens,
            full_sequence=full_sequence,
            input_length=input_length,
            generated_length=generated_length,
            hidden_states=hidden_states,
            scores=scores,
            finish_reason=finish_reason,
            token_ids=token_ids,
            keep_indices=keep_indices,
        )

    def _determine_finish_reason(
        self,
        generated_tokens: torch.Tensor,
        generated_length: int,
        max_tokens: int,
    ) -> str:
        if generated_length == 0:
            return "empty"
        last_token = int(generated_tokens[0, -1].item())
        if last_token in self._terminator_ids:
            return "eos"
        if generated_length >= max_tokens:
            return "length"
        return "other"

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_special_ids(self) -> Set[int]:
        return set(getattr(self._tokenizer, "all_special_ids", []) or [])

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"ModelRunner('{self.model_name_or_path}', {status})"