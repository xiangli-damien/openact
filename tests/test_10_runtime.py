"""Offline numerical and storage checks using real, tiny Transformers models."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
    PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
)

from openact_collect import CollectionRunner, GenericTask, ModelRunner
from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tracing.module_resolver import find_final_norm
from openact_core import Run


def make_runner(family='llama', dtype=torch.float32, attention='eager'):
    torch.manual_seed(7)
    config = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                  max_position_embeddings=256, eos_token_id=None, pad_token_id=0)
    if family in ('llama3', 'llama32'):
        config['rope_theta'] = 500000.0
    if family == 'llama32':
        config.update(max_position_embeddings=131072, tie_word_embeddings=True,
                      rope_scaling={'rope_type': 'llama3', 'factor': 32.0,
                                    'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
                                    'original_max_position_embeddings': 8192})
    if family == 'qwen2':
        config['rope_theta'] = 1000000.0
    if family == 'gpt2':
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=2,
                                         n_positions=256, eos_token_id=None, pad_token_id=0))
    elif family == 'qwen2':
        model = Qwen2ForCausalLM(Qwen2Config(**config))
    else:
        model = LlamaForCausalLM(LlamaConfig(**config))
    model.set_attn_implementation(attention)
    model = model.to(dtype).eval()
    backend = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'hello': 2, 'world': 3,
                                  'answer': 4, '2': 5, '+': 6, '?': 7}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')
    runner = ModelRunner('tiny-local', dtype=str(dtype).split('.')[-1])
    runner._model, runner._config, runner._tokenizer = model, model.config, tokenizer
    runner._loaded = True
    runner._probed_n_layers, runner._decoder_n_layers = 3, 2
    return runner


@pytest.mark.parametrize('family', ['llama', 'qwen2', 'gpt2'])
@pytest.mark.parametrize('n_tokens', [1, 4])
def test_teacher_forced_states_match_reference(family, n_tokens):
    runner = make_runner(family)
    spec = CaptureSpec(hidden_states_layers=[-1, 0], hidden_states_dtype='float32')
    prompt = torch.tensor([[2, 3, 5]])
    calls = []
    hook = runner.model.register_forward_pre_hook(lambda m, args, kw: calls.append(kw.get('output_hidden_states')), with_kwargs=True)
    result = runner.generate(prompt, GenerationSpec(max_new_tokens=n_tokens), capture_spec=spec)
    hook.remove()
    assert calls[-1] is True and not any(calls[:-1])
    assert result.keep_indices == list(range(1, n_tokens + 1))
    data = HiddenStateExtractor(spec, runner).extract(result, prompt)
    norm_inputs, norm_outputs = [], []
    hook = find_final_norm(runner.model).register_forward_hook(
        lambda m, args, output: (norm_inputs.append(args[0].detach().clone()), norm_outputs.append(output.detach().clone())) and None
    )
    with torch.inference_mode():
        reference = runner.model(result.full_sequence, output_hidden_states=True, use_cache=False)
    hook.remove()
    expected = torch.stack([reference.hidden_states[-1], reference.hidden_states[0]], dim=2)[0, 3:].numpy()
    np.testing.assert_allclose(data.per_token_states, expected, atol=1e-6)
    np.testing.assert_allclose(data.final_norm_pre, norm_inputs[0][0, 3:].numpy(), atol=1e-6)
    np.testing.assert_allclose(data.final_norm_post, norm_outputs[0][0, 3:].numpy(), atol=1e-6)
    np.testing.assert_allclose(data.final_norm_post, data.per_token_states[:, 0], atol=1e-6)
    np.testing.assert_allclose(data.final_norm_pre_mean, data.final_norm_pre.mean(0), atol=1e-6)
    np.testing.assert_allclose(data.final_norm_pre_prompt_last, norm_inputs[0][0, 2].numpy(), atol=1e-6)
    assert not np.allclose(data.final_norm_pre, data.final_norm_post)


@pytest.mark.parametrize('family', ['llama', 'qwen2'])
def test_bfloat16_and_optional_traces(family):
    runner = make_runner(family, torch.bfloat16)
    spec = CaptureSpec(attention=True, attention_save_patterns=True, attention_pattern_window=4,
                       mlp=True, hidden_states_dtype='float32')
    result = runner.generate(torch.tensor([[2, 3, 5]]), GenerationSpec(max_new_tokens=3), capture_spec=spec)
    data = HiddenStateExtractor(spec, runner).extract(result)
    assert data.attention_outputs.shape == data.mlp_outputs.shape == (3, 2, 16)
    assert data.attention_patterns.shape == (3, 2, 2, 4)
    assert np.isfinite(data.final_norm_pre).all()
    assert np.any(data.mlp_outputs)


def test_model_loading_and_probe(tmp_path):
    runner = make_runner()
    runner.model.save_pretrained(tmp_path)
    runner.tokenizer.save_pretrained(tmp_path)
    loaded = ModelRunner(str(tmp_path), device_map='cpu', dtype='float32')
    loaded.load()
    assert loaded.probed_n_layers == 3
    assert loaded.decoder_n_layers == 2
    assert loaded.apply_chat_template([{'role': 'user', 'content': 'hello world'}]).shape == (1, 2)


def test_collection_roundtrip_and_refuse_overwrite(tmp_path):
    runner = make_runner()
    task = GenericTask([{'prompt': 'hello world', 'answer': '2'}, {'prompt': 'hello', 'answer': '3'}])
    spec = CaptureSpec(hidden_states_layers=[-1], hidden_states_dtype='float32')
    collector = CollectionRunner(runner, task, tmp_path, capture_spec=spec, generation_spec=GenerationSpec(max_new_tokens=3))
    stats = collector.run()
    assert stats['ok'] == 2
    run = Run(tmp_path)
    assert run.validate() == []
    assert run.is_complete
    assert run.manifest.custom['activation_extraction'] == 'teacher_forced_forward'
    assert run.manifest.prompt.chat_template_applied is False
    for sample in run:
        assert sample.n_tokens == 3
        assert sample.get_final_norm_states('pre').shape == (3, 16)
        np.testing.assert_allclose(sample.hidden_states[:, 0], sample.get_final_norm_states('post'))
        assert sample.get_final_norm_states('pre', 'mean').shape == (16,)
    with pytest.raises(FileExistsError):
        collector.run()


@pytest.mark.parametrize('family', ['llama2', 'llama3', 'llama32', 'qwen2'])
@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
def test_paper_all_layers_match_independent_causal_prefixes(tmp_path, family, attention):
    from scripts.check_paper_collection import verify_saved_sample
    runner = make_runner(family, attention=attention)
    # A real chat wrapper makes prompt-last distinct from the last content word.
    runner.tokenizer.chat_template = "{{ 'hello ' + messages[0]['content'] + ' answer' }}"
    task = GenericTask([{'prompt': 'hello world', 'answer': '2'}])
    capture = CaptureSpec(hidden_states_dtype='float32')
    CollectionRunner(runner, task, tmp_path, capture_spec=capture,
                     generation_spec=GenerationSpec(max_new_tokens=4)).run()
    sample = Run(tmp_path)[0]
    assert sample.prompt_token_ids.tolist() == [2, 2, 3, 4]
    checks = verify_saved_sample(runner, sample, tolerance=1e-5)
    assert checks['hidden_shape'] == [4, 3, 16]
    assert checks['prefix_forwards_checked'] == 5
    # Prefix average is recoverable using only states up to the chosen boundary.
    prefix_ids = torch.tensor([sample.prompt_token_ids.tolist() + sample.token_ids[:2].tolist()])
    with torch.inference_mode():
        prefix = runner.model(prefix_ids, output_hidden_states=True, use_cache=False)
    expected = torch.stack([h[0, -2:].float().mean(0) for h in prefix.hidden_states]).numpy()
    np.testing.assert_allclose(sample.hidden_states[:2].mean(0), expected, atol=1e-6)


def test_non_eos_special_tokens_are_not_terminators():
    runner = make_runner()
    runner._model.generation_config.eos_token_id = [5, 6]
    runner._tokenizer = SimpleNamespace(eos_token_id=5, additional_special_tokens_ids=[7, 8, 9])
    assert runner._get_termination_tokens() == [5, 6]


def test_default_generation_and_extractor_settings_agree():
    runner = make_runner()
    result = runner.generate(torch.tensor([[2, 3]]), GenerationSpec(max_new_tokens=2))
    data = HiddenStateExtractor(CaptureSpec(), runner).extract(result)
    assert data.final_norm_pre.shape == (2, 16)


def test_eos_is_kept_and_has_its_own_state():
    runner = make_runner()
    def force_eos(input_ids, scores):
        scores.fill_(-float('inf'))
        scores[:, 5] = 0
        return scores
    runner._terminator_ids = [5]
    result = runner.generate(torch.tensor([[2, 3]]), GenerationSpec(max_new_tokens=4), logits_processors=[force_eos], capture_spec=CaptureSpec())
    assert result.token_ids == [5]
    assert result.finish_reason == 'eos'
    assert result.keep_indices == [1]
    assert len(result.traces.final_norm_pre) == 2
    data = HiddenStateExtractor(CaptureSpec(), runner).extract(result)
    with torch.inference_mode():
        expected = runner.model(result.full_sequence, output_hidden_states=True, use_cache=False)
    np.testing.assert_allclose(data.per_token_states[0, -1], expected.hidden_states[-1][0, -1].numpy(), rtol=1e-3, atol=1e-3)


def test_no_hidden_states_still_aligns_traces():
    runner = make_runner()
    spec = CaptureSpec(hidden_states=False, attention=True, mlp=True)
    result = runner.generate(torch.tensor([[2, 3]]), GenerationSpec(max_new_tokens=2), capture_spec=spec)
    data = HiddenStateExtractor(spec, runner).extract(result)
    assert data.per_token_states is None
    assert data.attention_outputs.shape == (2, 2, 16)


def test_pattern_capture_rejects_sdpa():
    runner = make_runner(attention='sdpa')
    with pytest.raises(ValueError, match='eager'):
        runner.generate(torch.tensor([[2, 3]]), GenerationSpec(max_new_tokens=2), capture_spec=CaptureSpec(attention=True, attention_save_patterns=True))


@pytest.mark.parametrize('layers', [[99], [-4], [], [0, 0]])
def test_invalid_layers_fail(layers):
    with pytest.raises(ValueError):
        CaptureSpec(hidden_states_layers=layers).get_effective_layers(3)
