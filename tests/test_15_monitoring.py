import json

from scripts.monitor_collections import (
    GIB, PublishedShards, advance, log_snapshot, meminfo, pressure, sample_job, transitions,
)


def test_memory_uses_available_and_gpu_alert_requires_sustained_pressure():
    memory = meminfo('MemTotal: 200000000 kB\nMemFree: 1000 kB\n'
                     'MemAvailable: 180000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n')
    resources = {'memory': memory, 'gpu': {'used_mib': 39000, 'total_mib': 40960},
                 'disk_free_bytes': 400 * GIB}
    state = {}
    assert pressure(resources, state, 0) == {}
    assert pressure(resources, state, 59) == {}
    assert pressure(resources, state, 60) == {'gpu_memory': 'critical'}
    resources['gpu']['used_mib'] = 7000
    resources['memory']['MemAvailable'] = 7 * GIB
    resources['disk_free_bytes'] = 60 * GIB
    assert pressure(resources, state, 90) == {'system_memory': 'critical', 'local_disk': 'critical'}
    resources['gpu']['used_mib'] = 39000
    assert 'gpu_memory' not in pressure(resources, state, 100)


def test_expected_waiting_does_not_accumulate_stall_time():
    state = {}
    assert advance(state, ['same'], 0) == 0
    assert advance(state, ['same'], 601) == 601
    assert advance(state, ['same'], 700, waiting=True) == 0
    assert advance(state, ['same'], 710) == 10
    assert advance(state, ['new'], 720) == 0


def test_log_incremental_errors_and_progress_reset_on_next_shard(tmp_path):
    path = tmp_path / 'job.log'
    path.write_text('\rCollecting: 99%|x| 99/100 [OK=100, Err=0]\n'
                    'PUBLISHED llama32\n\rCollecting: 0%| | 0/100 [00:00<?, ?it/s]')
    state = {}
    log, errors = log_snapshot(path, state)
    assert log['current_bar']['done'] == 0
    assert errors == []
    with path.open('a') as f:
        f.write('\nCUDA out of memory\nTraceback (most recent call last):\n'
                '\rCollecting: 0%| | 0/100 [OK=1, Err=0]')
    log, errors = log_snapshot(path, state)
    assert log['current_bar']['done'] == 1
    assert len(errors) == 2
    assert log_snapshot(path, state)[1] == []
    path.write_text('Transfer retry 1: example\n')
    assert len(log_snapshot(path, state)[1]) == 1


def test_alerts_only_emit_on_change_or_recovery():
    assert transitions({'gpu': 'warning'}, {'gpu': 'warning'}) == []
    assert transitions({'gpu': 'warning'}, {'gpu': 'critical'}) == [
        {'kind': 'alert', 'key': 'gpu', 'severity': 'critical'}]
    assert transitions({'gpu': 'critical'}, {}) == [{'kind': 'recovery', 'key': 'gpu'}]


def test_published_receipt_beats_lagging_status_and_missing_process_is_detected(tmp_path, monkeypatch):
    root, local, logs = tmp_path / 'math_full_20260916', tmp_path / 'staging', tmp_path / 'logs'
    shard = root / 'llama32' / 'shard_00000_00100'
    shard.mkdir(parents=True)
    local.mkdir()
    logs.mkdir()
    (shard / '_SHARD.json').write_text(json.dumps({
        'passed': True, 'fingerprint': 'test', 'model_alias': 'llama32',
        'samples': 100, 'start': 0, 'stop': 100, 'tokens': 1234,
        'evaluation': {'n_error': 0, 'n_evaluated': 100}}))
    (shard / '_COPY_VERIFIED.json').write_text(json.dumps({
        'stored_bytes': 5678, 'files': {'labels/correctness.parquet': {}}}))
    (shard / '_SUCCESS').write_text('{}')
    (root / 'job_status.json').write_text(json.dumps({
        'fingerprint': 'test', 'pid': 123, 'local_root': str(local),
        'completed_samples': 0, 'status': 'running', 'errors': [], 'expected_total': 15000}))
    (logs / (root.name + '.log')).write_text('Collecting: 100%|x| 100/100 [OK=100, Err=0]')
    monkeypatch.setattr('scripts.monitor_collections.process', lambda *a: {
        'alive': False, 'state': 'missing', 'wchar': 0})
    cache = PublishedShards()
    job, alerts, _ = sample_job(root, logs, cache, {}, 100)
    assert job['published_samples'] == 100
    assert job['published']['llama32']['tokens'] == 1234
    assert alerts == {'process_missing': 'critical'}
    (shard / '_SHARD.json').write_text('{"passed": false}')
    job, alerts, _ = sample_job(root, logs, cache, {}, 130)
    assert job['published_samples'] == 0
    assert alerts['published_metadata'] == 'critical'
