import numpy as np
import torch
from openact_collect import GenericTask
from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_core import Run
from scripts.run_safety_balanced import BatchMetrics, ReplayCollection, capture_saved, quota_selection, trim_response
from tests.test_10_runtime import make_runner


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
