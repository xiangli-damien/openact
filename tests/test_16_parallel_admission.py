from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from scripts.benchmark_live_parallel import assessment, select_window
from scripts.run_parallel_mmlu_collection import pair_admitted, handoff_ready, active_lease


def example():
    math = {'sample_ids': ['math1'], 'token_ids': [[1, 2]], 'collect_seconds': 100}
    mmlu = {'sample_ids': ['mmlu1'], 'token_ids': [[3, 4]], 'collect_seconds': 50}
    pair_math = {**math, 'started': 0, 'collection_finished': 120, 'passed': True}
    pair_mmlu = {**mmlu, 'started': .05, 'collection_finished': 80, 'passed': True}
    return [math, mmlu, pair_math, pair_mmlu, 35000, 3400]


def test_admission_is_exact_and_accounts_for_extra_paused_worker():
    result = assessment(*example())
    assert result['combined_speedup'] == 1.25
    assert result['projected_headroom_mib'] == 9360
    assert result['eligible'] and pair_admitted(result)
    assert not pair_admitted({'eligible': True})
    for field, value in [('tokens_identical', False), ('activations_valid', False),
                         ('combined_speedup', 1.04), ('projected_headroom_mib', 8191),
                         ('status', 'failed')]:
        changed = {**result, field: value}
        assert not pair_admitted(changed)


def test_tokens_and_workload_must_match_and_time_must_include_both_workers():
    args = example()
    args[2]['token_ids'] = [[1, 99]]
    assert not assessment(*args)['eligible']
    args = example()
    args[3]['sample_ids'] = ['different_question']
    assert not assessment(*args)['eligible']
    args = example()
    args[3]['collection_finished'] = 160
    assert not assessment(*args)['eligible']
    args = example()
    args[4] = 36500
    assert not assessment(*args)['eligible']


def test_window_skips_inflight_request_and_does_not_cross_shard():
    assert select_window(30, 100, 6) == (31, 37)
    with pytest.raises(ValueError):
        select_window(93, 100, 6)


def test_handoff_only_after_idle_large_model_gate():
    base = {'status': 'running', 'current_model': 'qwen2', 'stage': 'waiting_for_math'}
    assert handoff_ready(base)
    assert not handoff_ready({**base, 'stage': 'collecting'})
    assert not handoff_ready({**base, 'current_model': 'llama32'})
    assert not handoff_ready({**base, 'status': 'failed'})


def test_dead_or_expired_pause_cannot_block_collection(tmp_path):
    path = tmp_path / 'lease.json'
    path.write_text(json.dumps({'expires_at': 200, 'controller_pid': os.getpid()}))
    assert active_lease(path, at=100)
    assert not active_lease(path, at=201)
    path.write_text(json.dumps({'expires_at': 200, 'controller_pid': 99999999}))
    assert not active_lease(path, at=100)


def test_full_checksum_verification_is_not_skipped_on_resume(tmp_path, monkeypatch):
    from scripts import run_parallel_mmlu_collection as adapter
    shard = tmp_path / 'qwen2' / 'shard_00000_00100'
    shard.mkdir(parents=True)
    (shard / '_COPY_VERIFIED.json').write_text('{"stored_bytes": 123}')
    calls = []
    def verify(path):
        calls.append(path)
        return json.loads((path / '_COPY_VERIFIED.json').read_text())
    monkeypatch.setattr(adapter.legacy, 'verified_receipt', verify)
    cached = adapter.checked_receipts(tmp_path)
    assert len(calls) == 1
    assert cached(shard)['stored_bytes'] == 123
    assert len(calls) == 1
    cached(shard)
    assert len(calls) == 2


def test_expired_or_missing_maintenance_controller_is_not_ignored(tmp_path):
    from scripts.monitor_collections import maintenance
    path = tmp_path / 'maintenance_lease.json'
    path.write_text(json.dumps({'expires_at': 200, 'controller_pid': 99999999}))
    assert maintenance(tmp_path, 100) is None
    path.write_text(json.dumps({'expires_at': 99, 'controller_pid': os.getpid()}))
    assert maintenance(tmp_path, 100) is None


def test_large_gate_requires_pinned_matching_evidence(tmp_path):
    from scripts.run_parallel_mmlu_collection import ParallelGate
    from scripts.run_math_collection import sha256
    primary, output = tmp_path / 'math', tmp_path / 'mmlu'
    primary.mkdir()
    output.mkdir()
    for root, fingerprint in ((primary, 'math-fp'), (output, 'mmlu-fp')):
        (root / 'job_plan.json').write_text(json.dumps({'fingerprint': fingerprint}))
    report = tmp_path / 'report.json'
    record = {**assessment(*example()), 'math_model': 'qwen2', 'mmlu_model': 'qwen2',
              'math_fingerprint': 'math-fp', 'mmlu_fingerprint': 'mmlu-fp'}
    report.write_text(json.dumps(record))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'pairs': {'qwen2:qwen2': {'report': str(report), 'sha256': sha256(report)}}}))
    gate = ParallelGate(primary, {'comparisons': {}}, 'qwen2', lambda *args: None)
    gate.policy_path, gate.output = policy, output
    assert pair_admitted(gate.evidence('qwen2'))
    assert gate.evidence('llama3') is None
    gate.cached.clear()
    report.write_text(json.dumps({**record, 'math_fingerprint': 'different'}))
    assert gate.evidence('qwen2') is None
    policy.write_text(json.dumps({'pairs': {'qwen2:qwen2': {'report': str(report), 'sha256': sha256(report)}}}))
    assert gate.evidence('qwen2') is None


def test_large_gate_yields_on_pressure_then_reloads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_parallel_mmlu_collection as adapter
    primary = tmp_path / 'math'
    primary.mkdir()
    (primary / 'job_status.json').write_text(json.dumps({'status': 'running', 'current_model': 'qwen2'}))
    gate = adapter.ParallelGate(primary, {'comparisons': {}}, 'qwen2', lambda *args: None)
    gate.output = tmp_path
    monkeypatch.setattr(gate, 'evidence', lambda _: {**assessment(*example()), 'mmlu_peak_process_mib': 16000})
    memory = iter([(35000, 40960), (16000, 40960)])
    monkeypatch.setattr(adapter, 'gpu_memory', lambda: next(memory))
    monkeypatch.setattr(adapter.time, 'sleep', lambda _: None)
    runner = SimpleNamespace(_loaded=True)
    calls = []
    def unload():
        runner._loaded = False
        calls.append('unload')
    def load():
        runner._loaded = True
        calls.append('load')
    runner.load, runner.unload = load, unload
    gate.wait(runner)
    assert calls == ['unload', 'load']


def test_monitor_detects_scheduler_failure_without_interrupting_collectors(tmp_path, monkeypatch):
    from scripts import monitor_collections as monitor
    folder = tmp_path / 'runs/parallel_admission_test'
    folder.mkdir(parents=True)
    record = {'pid': 1, 'handoff_pid': 2, 'status': 'running', 'attempts': {}}
    path = folder / 'supervisor.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(monitor, 'process', lambda pid, path: {'pid': pid, 'alive': pid == 2})
    _, alerts = monitor.parallel_scheduler(tmp_path, False)
    assert alerts == {'parallel/supervisor_missing': 'warning'}
    path.write_text(json.dumps({**record, 'status': 'math_complete'}))
    assert monitor.parallel_scheduler(tmp_path, False)[1] == {}
    monkeypatch.setattr(monitor, 'process', lambda pid, path: {'pid': pid, 'alive': False})
    assert monitor.parallel_scheduler(tmp_path, False)[1] == {'parallel/handoff_missing': 'critical'}
    assert monitor.parallel_scheduler(tmp_path, True)[1] == {}
