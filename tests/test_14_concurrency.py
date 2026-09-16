from copy import deepcopy

from scripts.benchmark_collection_concurrency import assess


def cases():
    math = {'dataset': 'math', 'sample_ids': ['math_1'], 'token_ids': [[1, 2]]}
    mmlu = {'dataset': 'mmlu', 'sample_ids': ['mmlu_1'], 'token_ids': [[3, 4]]}
    return ({'wall_seconds': 100, 'workers': [math]},
            {'wall_seconds': 50, 'workers': [mmlu]},
            {'wall_seconds': 110, 'workers': [deepcopy(math), deepcopy(mmlu)],
             'peak_total_gpu_mib': 24000})


def test_parallel_gate_requires_real_combined_throughput_gain():
    math, mmlu, pair = cases()
    assert assess(math, mmlu, pair)['eligible']
    pair['wall_seconds'] = 155
    assert not assess(math, mmlu, pair)['eligible']


def test_parallel_gate_requires_identical_tokens_and_sample_ids():
    math, mmlu, pair = cases()
    pair['workers'][1]['token_ids'] = [[3, 5]]
    assert not assess(math, mmlu, pair)['eligible']
    math, mmlu, pair = cases()
    pair['workers'][0]['sample_ids'] = ['wrong_question']
    assert not assess(math, mmlu, pair)['eligible']


def test_parallel_gate_keeps_eight_gib_headroom():
    math, mmlu, pair = cases()
    pair['peak_total_gpu_mib'] = 35000
    assert not assess(math, mmlu, pair)['eligible']


def test_mmlu_uses_three_full_datasets():
    from pathlib import Path
    from openact_collect.config import tomllib
    from scripts.run_mmlu_collection import validate_scope
    path = Path(__file__).resolve().parents[1] / 'configs/mmlu_full.toml'
    config = tomllib.loads(path.read_text())
    validate_scope(config)
    assert len(config['models']) * config['datasets'][0]['expected_samples'] == 42126


def test_large_mmlu_models_wait_until_math_finishes():
    from scripts.run_mmlu_collection import may_run
    comparisons = {alias: {'eligible': True} for alias in ('llama32', 'qwen2', 'llama3')}
    for math_model in comparisons:
        state = {'status': 'running', 'current_model': math_model}
        assert may_run('llama32', False, state, comparisons)
        for model in ('qwen2', 'llama3'):
            assert not may_run(model, False, state, comparisons)
            assert may_run(model, True, {'status': 'complete'}, comparisons)


def test_small_mmlu_yields_for_unapproved_or_failed_primary():
    from scripts.run_mmlu_collection import may_run
    comparisons = {'llama32': {'eligible': True}, 'qwen2': {'eligible': False}}
    assert not may_run('llama32', False, {'status': 'running', 'current_model': 'qwen2'}, comparisons)
    assert not may_run('llama32', False, {'status': 'failed', 'current_model': 'llama32'}, comparisons)


def test_gate_releases_gpu_then_reloads_after_math_completion(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import run_mmlu_collection as mmlu
    primary = tmp_path / 'primary'
    primary.mkdir()
    status = primary / 'job_status.json'
    status.write_text(json.dumps({'status': 'running', 'current_model': 'qwen2'}))
    calls = []
    runner = SimpleNamespace(_loaded=True)
    def unload():
        runner._loaded = False
        calls.append('unload')
    def load():
        runner._loaded = True
        calls.append('load')
    def sleep(_):
        status.write_text(json.dumps({'status': 'complete'}))
        (primary / '_SUCCESS').touch()
    runner.load, runner.unload = load, unload
    monkeypatch.setattr(mmlu.time, 'sleep', sleep)
    gate = mmlu.MathPriorityGate(primary, {'comparisons': {}}, 'qwen2', lambda *x: None)
    gate.wait(runner)
    assert calls == ['unload', 'load']


def test_failed_memory_or_correctness_cannot_approve_production():
    import pytest
    from scripts.run_mmlu_collection import validate_benchmark
    valid = {'status': 'complete', 'comparisons': {
        alias: {'tokens_identical': True, 'memory_headroom_mib': 12000, 'eligible': True}
        for alias in ('llama32', 'qwen2', 'llama3')
    }}
    validate_benchmark(valid)
    valid['comparisons']['llama3']['memory_headroom_mib'] = 4000
    with pytest.raises(ValueError, match='headroom'):
        validate_benchmark(valid)


def test_mmlu_shard_collect_verify_evaluate_publish(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pandas as pd
    from openact_collect.schema import CaptureSpec, GenerationSpec
    from openact_collect.tasks.prepared import PreparedParquetTask
    from openact_core import Run
    from scripts import run_mmlu_collection as mmlu
    from tests.test_10_runtime import make_runner
    runner = make_runner()
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    pd.DataFrame([
        {'sample_id': 'mmlu_a', 'prompt_text': 'hello world', 'ground_truth': 'A', 'subject': 'algebra'},
        {'sample_id': 'mmlu_b', 'prompt_text': 'hello', 'ground_truth': 'B', 'subject': 'algebra'},
    ]).to_parquet(prepared / 'part_00000.parquet', index=False)
    (prepared / 'prepared_manifest.json').write_text(json.dumps({
        'task': 'mmlu', 'task_source': 'test', 'split': 'test', 'prompt_template_variant': 'zot',
    }))
    items = list(PreparedParquetTask(prepared).iter_items())
    admissions = []
    gate = SimpleNamespace(wait=lambda r: admissions.append(r))
    monkeypatch.setattr(mmlu.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(mmlu.torch.cuda, 'max_memory_allocated', lambda: 0)
    local, destination = tmp_path / 'local', tmp_path / 'persistent'
    mmlu.collect_shard(runner, gate, prepared, items, local,
                       {'model': {}, 'collection': {}}, GenerationSpec(max_new_tokens=3),
                       CaptureSpec(hidden_states_dtype='float32'),
                       {'dataset': 'mmlu', 'model_alias': 'llama32', 'start': 0, 'stop': 2})
    assert len(admissions) == 4  # Before shard, each sample, and verification.
    record = json.loads((local / '_SHARD.json').read_text())
    assert record['evaluation']['n_evaluated'] == 2
    assert record['evaluation']['n_error'] == 0
    assert record['evaluation']['evaluator'] == 'parser/mmlu'
    assert record['verification']['fixed_shape_max_relative_l2_error'] == 0
    mmlu.publish_shard(local, destination)
    run = Run(destination, require_complete=True)
    assert [s.sample_id for s in run] == ['mmlu_a', 'mmlu_b']
    assert (destination / 'labels/correctness.parquet').exists()
    assert not local.exists()
