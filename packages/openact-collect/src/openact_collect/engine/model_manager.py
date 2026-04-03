import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessorList
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tracing import ActivationRecorder, ActivationTrace
from openact_collect.tracing.module_resolver import find_transformer_layers
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
    traces: Optional[ActivationTrace] = None


class ModelRunner:
    def __init__(self, model_name_or_path: str, device_map: str = 'auto', dtype: str = 'auto', trust_remote_code: bool = True, attn_implementation: Optional[str] = None):
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
        self._decoder_n_layers: Optional[int] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._config = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=self.trust_remote_code)
        model_kwargs: Dict[str, Any] = {
            'config': self._config,
            'torch_dtype': self._resolve_dtype(self.dtype_str),
            'device_map': self.device_map,
            'trust_remote_code': self.trust_remote_code,
        }
        if self.attn_implementation:
            model_kwargs['attn_implementation'] = self.attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=self.trust_remote_code)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._terminator_ids = self._get_termination_tokens()
        self._probed_n_layers = self._probe_n_layers()
        self._decoder_n_layers = self._probe_decoder_n_layers()
        self._loaded = True

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        if dtype == 'auto':
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        return {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}.get(dtype, torch.float16)

    def _get_termination_tokens(self) -> List[int]:
        terminators = [self._tokenizer.eos_token_id]
        for token_id in getattr(self._tokenizer, 'additional_special_tokens_ids', []) or []:
            if token_id not in terminators:
                terminators.append(token_id)
        for token in ('<|eot_id|>', '<|redacted_im_end|>', '</s>'):
            try:
                token_id = self._tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if token_id is not None and token_id != self._tokenizer.unk_token_id and token_id not in terminators:
                terminators.append(token_id)
        return terminators

    def _probe_n_layers(self) -> int:
        try:
            batch = self._tokenizer('probe', return_tensors='pt')
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self._model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
            if isinstance(output.hidden_states, (list, tuple)):
                return len(output.hidden_states)
        except Exception:
            logger.debug('Probe forward failed, falling back to config', exc_info=True)
        cfg = self._config
        base = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layer', None) or getattr(cfg, 'n_layers', None) or 0
        return int(base) + 1

    def _probe_decoder_n_layers(self) -> int:
        try:
            return len(find_transformer_layers(self.model))
        except Exception:
            cfg = self._config
            base = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layer', None) or getattr(cfg, 'n_layers', None) or 0
            return int(base)

    @property
    def model(self) -> PreTrainedModel:
        if not self._loaded:
            raise RuntimeError('Model not loaded. Call load() first.')
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if not self._loaded:
            raise RuntimeError('Model not loaded. Call load() first.')
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
            raise RuntimeError('Model not loaded. Call load() first.')
        return self._probed_n_layers

    @property
    def decoder_n_layers(self) -> int:
        if self._decoder_n_layers is None:
            raise RuntimeError('Model not loaded. Call load() first.')
        return self._decoder_n_layers

    def get_model_spec(self) -> ModelSpec:
        cfg = self._config
        return ModelSpec(
            name=Path(self.model_name_or_path).name,
            source='huggingface' if '/' in self.model_name_or_path else 'local',
            identifier=self.model_name_or_path,
            dtype=self.dtype_str,
            architecture=cfg.architectures[0] if getattr(cfg, 'architectures', None) else None,
            n_layers=self._probed_n_layers,
            n_decoder_layers=self._decoder_n_layers,
            hidden_dim=getattr(cfg, 'hidden_size', None) or getattr(cfg, 'n_embd', None) or getattr(cfg, 'd_model', None),
            n_heads=getattr(cfg, 'num_attention_heads', None) or getattr(cfg, 'n_head', None),
            vocab_size=getattr(cfg, 'vocab_size', None),
            tokenizer_class=type(self._tokenizer).__name__,
        )

    def render_chat_text(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            if isinstance(rendered, str):
                return rendered
            if isinstance(rendered, list):
                return ''.join(str(part) for part in rendered)
            return str(rendered)
        return '\n'.join(message['content'] for message in messages)

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> torch.Tensor:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            result = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors='pt')
            if hasattr(result, 'input_ids'):
                return result.input_ids
            if not isinstance(result, torch.Tensor):
                return torch.tensor([result]) if isinstance(result, list) else result
            return result
        text = '\n'.join(message['content'] for message in messages)
        return self.tokenizer(text, return_tensors='pt').input_ids

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _align_hidden_states(self, hidden_states: List[Tuple[torch.Tensor, ...]], generated_length: int) -> List[Tuple[torch.Tensor, ...]]:
        if not hidden_states:
            return hidden_states
        hidden_state_steps = len(hidden_states)
        if hidden_state_steps in (generated_length, generated_length + 1):
            return hidden_states
        if hidden_state_steps > generated_length + 1:
            warnings.warn(f'Hidden states length {hidden_state_steps} > generated_length+1 ({generated_length + 1}); keeping the tail.', UserWarning, stacklevel=2)
            return hidden_states[-(generated_length + 1):]
        warnings.warn(f'Hidden states length {hidden_state_steps} < generated_length {generated_length}; some tokens will be truncated.', UserWarning, stacklevel=2)
        return hidden_states

    @staticmethod
    def _build_keep_indices(hidden_states: List[Tuple[torch.Tensor, ...]], generated_ids: List[int], generated_length: int) -> Tuple[List[int], List[int]]:
        if not hidden_states or generated_length == 0:
            return [], []
        has_prompt_step = len(hidden_states) == generated_length + 1
        keep_indices: List[int] = []
        token_ids: List[int] = []
        for index, token_id in enumerate(generated_ids):
            step_idx = index + 1 if has_prompt_step else index
            if step_idx >= len(hidden_states):
                break
            keep_indices.append(step_idx)
            token_ids.append(token_id)
        return keep_indices, token_ids

    def generate(self, input_ids: torch.Tensor, gen_spec: GenerationSpec, output_scores: bool = False, logits_processors: Optional[List[Any]] = None, capture_hidden_states: Optional[bool] = None, capture_spec: Optional[CaptureSpec] = None) -> GenerationResult:
        self._set_seed(int(gen_spec.seed))
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        input_length = int(input_ids.shape[1])
        do_sample = bool(gen_spec.do_sample) and gen_spec.temperature > 0
        if capture_spec is not None:
            capture_hidden_states = bool(capture_spec.hidden_states) if capture_hidden_states is None else bool(capture_hidden_states)
            output_attentions = bool(capture_spec.attention and capture_spec.attention_save_patterns)
        else:
            capture_hidden_states = True if capture_hidden_states is None else bool(capture_hidden_states)
            output_attentions = False
        kwargs: Dict[str, Any] = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': int(gen_spec.max_new_tokens),
            'do_sample': do_sample,
            'eos_token_id': self._terminator_ids,
            'pad_token_id': self.tokenizer.pad_token_id,
            'return_dict_in_generate': True,
            'output_scores': output_scores,
            'output_hidden_states': bool(capture_hidden_states),
            'output_attentions': bool(output_attentions),
        }
        if do_sample:
            kwargs['temperature'] = float(gen_spec.temperature)
            kwargs['top_p'] = float(gen_spec.top_p)
            if gen_spec.top_k is not None:
                kwargs['top_k'] = int(gen_spec.top_k)
        if logits_processors:
            kwargs['logits_processor'] = LogitsProcessorList(logits_processors)
        traces: Optional[ActivationTrace] = None
        with torch.inference_mode():
            if capture_spec is not None:
                with ActivationRecorder(self.model, capture_spec, decoder_n_layers=self.decoder_n_layers) as recorder:
                    output = self.model.generate(**kwargs)
                    traces = recorder.trace()
            else:
                output = self.model.generate(**kwargs)
        if getattr(output, 'attentions', None) is not None:
            output.attentions = None
        sequences = output.sequences
        generated_tokens = sequences[:, input_length:]
        generated_ids = sequences[0, input_length:].tolist()
        generated_length = len(generated_ids)
        hidden_states = list(output.hidden_states) if capture_hidden_states and getattr(output, 'hidden_states', None) else []
        scores = list(output.scores) if output_scores and getattr(output, 'scores', None) else None
        if hidden_states:
            hidden_states = self._align_hidden_states(hidden_states, generated_length)
            keep_indices, token_ids = self._build_keep_indices(hidden_states, generated_ids, generated_length)
        else:
            keep_indices, token_ids = [], list(generated_ids)
        return GenerationResult(
            input_ids=input_ids,
            generated_tokens=generated_tokens,
            full_sequence=sequences,
            input_length=input_length,
            generated_length=generated_length,
            hidden_states=hidden_states,
            scores=scores,
            finish_reason=self._determine_finish_reason(generated_tokens, generated_length, gen_spec.max_new_tokens),
            token_ids=token_ids,
            keep_indices=keep_indices,
            traces=traces,
        )

    def _determine_finish_reason(self, generated_tokens: torch.Tensor, generated_length: int, max_tokens: int) -> str:
        if generated_length == 0:
            return 'empty'
        last_token = int(generated_tokens[0, -1].item())
        if last_token in self._terminator_ids:
            return 'eos'
        if generated_length >= max_tokens:
            return 'length'
        return 'other'

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_special_ids(self) -> Set[int]:
        return set(getattr(self._tokenizer, 'all_special_ids', []) or [])

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        state = 'loaded' if self._loaded else 'not loaded'
        return f"ModelRunner({self.model_name_or_path!r}, {state})"
