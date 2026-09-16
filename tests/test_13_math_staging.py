import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openact_collect.config import tomllib
from scripts import run_math_collection as staging


def make_shard(path):
    path.mkdir(parents=True)
    (path / 'tensors.zarr').mkdir()
    (path / 'tensors.zarr' / '0.0').write_bytes(bytes(range(256)) * 10)
    staging.atomic_json(path / '_SUCCESS', {'complete': True})
    staging.atomic_json(path / '_SHARD.json', {'samples': 1, 'sample_ids': ['math_0']})


def test_math_job_scope_is_exact_and_full():
    path = Path(__file__).resolve().parents[1] / 'configs/math_full.toml'
    config = tomllib.loads(path.read_text())
    staging.validate_scope(config)
    assert len(config['models']) * config['datasets'][0]['expected_samples'] == 15000
    config['datasets'].append({'key': 'mmlu', 'task': 'mmlu', 'expected_samples': 14042})
    with pytest.raises(ValueError, match='Only the full'):
        staging.validate_scope(config)


@pytest.mark.parametrize('total,size', [(5000, 100), (5000, 127), (3, 100), (1, 1)])
def test_shards_cover_every_row_once(total, size):
    ranges = staging.shard_ranges(total, size)
    assert [i for start, stop in ranges for i in range(start, stop)] == list(range(total))
    assert all(0 < stop - start <= size for start, stop in ranges)


def test_copy_verifies_all_files_before_deleting_local(tmp_path):
    local, destination = tmp_path / 'local', tmp_path / 'dami' / 'shard'
    make_shard(local)
    before = staging.inventory(local)
    result = staging.publish_shard(local, destination)
    assert not local.exists()
    assert staging.verified_receipt(destination)['files'] == before
    assert result['stored_bytes'] == sum(f['bytes'] for f in before.values())
    assert result['run_path'] == str(destination)


def test_corrupted_copy_keeps_local_and_never_publishes(tmp_path, monkeypatch):
    local, destination = tmp_path / 'local', tmp_path / 'dami' / 'shard'
    make_shard(local)
    copy = staging.durable_copy
    def corrupted(src, dst):
        result = copy(src, dst)
        if Path(src).name == '0.0':
            Path(dst).write_bytes(b'corrupt')
        return result
    monkeypatch.setattr(staging, 'durable_copy', corrupted)
    with pytest.raises(ValueError, match='checksum mismatch'):
        staging.publish_shard(local, destination)
    assert (local / 'tensors.zarr' / '0.0').read_bytes() == bytes(range(256)) * 10
    assert not destination.exists()


def test_idempotent_copy_checks_existing_destination(tmp_path):
    local, destination = tmp_path / 'local', tmp_path / 'dami' / 'shard'
    make_shard(local)
    staging.publish_shard(local, destination)
    make_shard(local)
    staging.publish_shard(local, destination)
    assert not local.exists()
    make_shard(local)
    (local / 'tensors.zarr' / '0.0').write_bytes(b'different local output')
    with pytest.raises(FileExistsError):
        staging.publish_shard(local, destination)
    assert local.exists()


def test_existing_copy_tampering_is_detected(tmp_path):
    local, destination = tmp_path / 'local', tmp_path / 'dami' / 'shard'
    make_shard(local)
    staging.publish_shard(local, destination)
    (destination / 'tensors.zarr' / '0.0').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        staging.verified_receipt(destination)


def test_unfinished_shard_cannot_be_copied(tmp_path):
    local, destination = tmp_path / 'local', tmp_path / 'dami' / 'shard'
    make_shard(local)
    (local / '_SHARD.json').unlink()
    with pytest.raises(ValueError, match='unverified/incomplete'):
        staging.publish_shard(local, destination)
    assert local.exists()
    assert not destination.exists()


def test_ordered_sample_ids_reject_omissions_duplicates_and_reordering():
    def run(ids):
        return [SimpleNamespace(sample_id=sample_id) for sample_id in ids]
    staging.check_ids(run(['a', 'b']), ['a', 'b'])
    for ids in (['a'], ['b', 'a'], ['a', 'a']):
        with pytest.raises(ValueError, match='sample IDs'):
            staging.check_ids(run(ids), ['a', 'b'])


def test_transient_copy_error_is_retried(tmp_path, monkeypatch):
    local, destination = tmp_path / 'local', tmp_path / 'dami' / 'shard'
    make_shard(local)
    publish = staging.publish_shard
    attempts = []
    def interrupted(src, dst):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError('transient network error')
        return publish(src, dst)
    monkeypatch.setattr(staging, 'publish_shard', interrupted)
    monkeypatch.setattr(staging.time, 'sleep', lambda _: None)
    staging.publish_with_retries(local, destination)
    assert len(attempts) == 2
    assert not local.exists()
    assert json.loads((destination / staging.RECEIPT).read_text())['files']
