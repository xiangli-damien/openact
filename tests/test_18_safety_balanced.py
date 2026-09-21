import numpy as np
import pytest
import torch
from openact_collect import GenericTask
from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_core import Run
from scripts.run_safety_balanced import BatchMetrics, ReplayCollection, capture_saved, quota_selection, trim_response
from tests.test_10_runtime import make_runner
from scripts.prepare_safety_benchmark import benchmark_rows
from scripts import run_safety_balanced as safety


@pytest.mark.parametrize('family', ['llama', 'qwen2'])
def test_prompt_only_storage_matches_full_replay_and_is_causal(family):
    runner = make_runner(family)
    full = CaptureSpec(hidden_states_dtype='float32')
    capture = safety.capture_spec_for_mode(full.to_dict(), 'prompt-last')
    prompt = [2, 3, 5]
    replay = capture_saved(runner, prompt, [4, 5, 6, 7], full, 'length', 4)
    expected = HiddenStateExtractor(full, runner).extract(replay)
    calls = []
    hook = runner.model.register_forward_pre_hook(
        lambda m, args, kwargs: calls.append(kwargs['input_ids'].shape[1]), with_kwargs=True)
    runner.model.generate = lambda **kwargs: (_ for _ in ()).throw(AssertionError('Must not generate'))
    actual = safety.capture_prompt_last(runner, prompt, [4, 5, 6, 7], capture)
    changed_future = safety.capture_prompt_last(runner, prompt, [7, 6, 5, 4], capture)
    hook.remove()
    assert calls == [len(prompt) + 4, len(prompt) + 4]
    assert actual.n_tokens == 0 and actual.per_token_states is None and actual.mean_states is None
    assert actual.final_norm_pre is None and actual.final_norm_post is None
    for key in ['prompt_last_states', 'final_norm_pre_prompt_last', 'final_norm_post_prompt_last']:
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
        np.testing.assert_array_equal(getattr(actual, key), getattr(changed_future, key))


def test_prompt_only_run_preserves_response_tokens_labels_and_validates(tmp_path):
    runner = make_runner()
    task = GenericTask([{'prompt': 'hello world'}])
    item = next(task.iter_items())
    prompt = task.render_prompt(item)
    ids = runner.apply_chat_template([{'role': 'user', 'content': prompt}])[0].tolist()
    full = CaptureSpec(hidden_states_dtype='float32')
    capture = safety.capture_spec_for_mode(full.to_dict(), 'prompt-last')
    result = runner.generate(torch.tensor([ids]), GenerationSpec(max_new_tokens=4), capture_spec=full)
    row = dict(sample_id=item.sample_id, prompt_text=prompt, prompt_token_ids=ids, token_ids=result.token_ids,
               response_text=runner.decode(result.token_ids), finish_reason=result.finish_reason,
               effective_max_new_tokens=4, model_input_text=prompt, screen_index=0,
               generation_metrics={'entropy': 1., 'perplexity': 2., 'max_probability': .5},
               judge={'is_safe': False})
    runner.model.generate = lambda **kwargs: (_ for _ in ()).throw(AssertionError('Must not generate'))
    safety.ReplayCollection(runner, task, tmp_path/'run', capture_spec=capture,
        capture_mode='prompt-last', generation_spec=GenerationSpec(max_new_tokens=4),
        responses={item.sample_id: row}).run()
    run = Run(tmp_path/'run')
    validation = safety.validate_prompt_shard(runner, run, [item.sample_id])
    assert validation['prompt_last_exact_replay'] and not run.validate()
    assert run[0].token_ids.tolist() == row['token_ids']
    assert run[0].response_text == row['response_text']
    assert run[0].prompt_last_hidden_states.shape == (3, 16)
    assert run.manifest.custom['activation_saved_positions'] == 'prompt_last_only'
    assert run._df.n_hidden_state_tokens.tolist() == [0]
    assert safety.label_result([row])[0].is_correct is False
    # Validation must detect corruption, not just check array presence.
    import zarr
    writable = zarr.open_group(str(tmp_path/'run'/'tensors.zarr'), mode='a')
    writable['hidden_states/prompt_last'][0, 0, 0] += 1
    with pytest.raises(AssertionError):
        safety.validate_prompt_shard(runner, Run(tmp_path/'run'), [item.sample_id])


def test_oom_halves_batch_without_skipping_ids_or_retaining_failed_tensors(monkeypatch):
    import weakref
    attempts, refs, events = [], [], []
    def generate(runner, items, indices, budget):
        assert all(ref() is None for ref in refs)
        attempts.append(list(indices))
        if len(items)>2:
            temporary=torch.zeros(10)
            refs.append(weakref.ref(temporary))
            raise torch.cuda.OutOfMemoryError('injected allocation failure')
        return [{'screen_index':i} for i in indices]
    monkeypatch.setattr(safety,'generate_batch',generate)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    rows=safety.generate_with_oom_recovery(None,list(range(8)),list(range(100,108)),2048,
                                          on_oom=lambda size,error:events.append(size))
    assert attempts==[list(range(100,108)),list(range(100,104)),[100,101]]
    assert rows==[{'screen_index':100},{'screen_index':101}]
    assert events==[8,4]


def test_single_sample_oom_is_explicit_failure(monkeypatch):
    import pytest
    def fail(*args):
        raise torch.cuda.OutOfMemoryError('injected')
    monkeypatch.setattr(safety,'generate_batch',fail)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    with pytest.raises(RuntimeError,match='batch size 1'):
        safety.generate_with_oom_recovery(None,[1],[0],2048)


def test_official_adversarial_subset_no_completion_or_duplicate_behavior():
    source = [
        dict(data_type='vanilla_harmful', vanilla='goal', adversarial='', completion='NO LEAK'),
        dict(data_type='adversarial_harmful', vanilla='goal', adversarial='variant one', completion='NO LEAK'),
        dict(data_type='adversarial_harmful', vanilla='goal', adversarial='variant two', completion='NO LEAK'),
        dict(data_type='adversarial_benign', vanilla='benign', adversarial='benign variant', completion='NO LEAK'),
    ]
    rows = benchmark_rows(source)
    assert len(rows) == 1
    assert rows[0]['prompt_text'] in ['variant one\n\nPlease reason step by step.', 'variant two\n\nPlease reason step by step.']
    assert 'NO LEAK' not in str(rows)
    assert rows[0]['category'] == 'adversarial_harmful'


def test_metrics_stop_at_each_samples_first_eos():
    metrics = BatchMetrics([3])
    logits = torch.tensor([[0., 1., 2., 3.], [3., 2., 1., 0.]])
    metrics(torch.tensor([[3, 2], [3, 1]]), logits)
    metrics(torch.tensor([[3, 2, 3], [3, 1, 0]]), logits * 2)
    metrics(torch.tensor([[3, 2, 3, 3], [3, 1, 0, 3]]), logits * 3)
    assert metrics.count.tolist() == [1, 2]
    p = logits[0].softmax(0)
    np.testing.assert_allclose(metrics.result()[0]['entropy'], -(p * p.log()).sum().item(), rtol=1e-6)


def test_quotas_ignore_unknown_and_keep_natural_order():
    rows = [{'sample_id': str(i), 'judge': {'is_safe': x}} for i, x in enumerate([True, False, None, True, False, False])]
    chosen, counts = quota_selection(rows, {'safe': 1, 'unsafe': 0}, 2)
    assert [r['sample_id'] for r in chosen] == ['0', '1', '4']
    assert counts == {'safe': 2, 'unsafe': 2}
    assert trim_response([7, 3, 0, 0], [3]) == [7, 3]
    assert trim_response([7, 2], [3]) == [7, 2]


def test_replay_matches_existing_collection_without_generate():
    runner = make_runner()
    capture = CaptureSpec(hidden_states_dtype='float32')
    prompt = torch.tensor([[2, 3, 5]])
    original = runner.generate(prompt, GenerationSpec(max_new_tokens=4), capture_spec=capture)
    expected = HiddenStateExtractor(capture, runner).extract(original)
    runner.model.generate = lambda **kwargs: (_ for _ in ()).throw(AssertionError('Replay must not generate'))
    replay = capture_saved(runner, prompt[0].tolist(), original.token_ids, capture, original.finish_reason, 4)
    actual = HiddenStateExtractor(capture, runner).extract(replay)
    for key in ['per_token_states', 'mean_states', 'prompt_last_states', 'final_norm_pre', 'final_norm_post',
                'final_norm_pre_prompt_last', 'final_norm_post_prompt_last']:
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))


def test_replay_writes_normal_openact_run(tmp_path):
    runner = make_runner()
    task = GenericTask([{'prompt': 'hello world'}])
    item = next(task.iter_items())
    prompt_text = task.render_prompt(item)
    ids = runner.apply_chat_template([{'role': 'user', 'content': prompt_text}])[0].tolist()
    capture = CaptureSpec(hidden_states_dtype='float32')
    result = runner.generate(torch.tensor([ids]), GenerationSpec(max_new_tokens=4), capture_spec=capture)
    row = dict(prompt_text=prompt_text, prompt_token_ids=ids, token_ids=result.token_ids,
               response_text=runner.decode(result.token_ids), finish_reason=result.finish_reason,
               effective_max_new_tokens=4, model_input_text=prompt_text, screen_index=0,
               generation_metrics={'entropy': 1., 'perplexity': 2., 'max_probability': .5})
    stats = ReplayCollection(runner, task, tmp_path/'run', capture_spec=capture,
               generation_spec=GenerationSpec(max_new_tokens=4), responses={item.sample_id:row}).run()
    run = Run(tmp_path/'run')
    assert stats['ok'] == 1 and run.is_complete and not run.validate()
    assert run[0].token_ids.tolist() == result.token_ids
    assert run[0].hidden_states.shape == (4, 3, 16)
    assert run[0].get_final_norm_states('pre').shape == (4, 16)
