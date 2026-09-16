"""Full HF MATH collection with bounded local shards and verified NFS copies.

Each published shard is a normal OpenAct Run. Repeating the same command resumes
at shard boundaries; partial attempts are retained separately for diagnosis.
No other task or model can be selected by this launcher.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import uuid

import numpy as np
import torch

from openact_collect import CollectionRunner, ModelRunner
from openact_collect.config import load_run_config, tomllib
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_core import Run
from openact_eval.evaluators.registry import auto_select_evaluator

try:
    from scripts.run_collection_matrix import (
        SelectedPreparedTask, context_preflight, evaluate_run, verify_run,
    )
except ModuleNotFoundError:
    from run_collection_matrix import (
        SelectedPreparedTask, context_preflight, evaluate_run, verify_run,
    )


MODELS = {
    'llama32': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen2': 'Qwen/Qwen2-7B-Instruct',
    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
}
RECEIPT = '_COPY_VERIFIED.json'
GIB = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('w') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(path):
    result = {}
    for file in sorted(Path(path).rglob('*')):
        if file.is_symlink():
            raise ValueError(f'Symlink not allowed in a collection shard: {file}')
        if file.is_file() and file.relative_to(path).as_posix() != RECEIPT:
            result[file.relative_to(path).as_posix()] = {
                'bytes': file.stat().st_size, 'sha256': sha256(file),
            }
    return result


def verified_receipt(path):
    receipt = json.loads((Path(path) / RECEIPT).read_text())
    if inventory(path) != receipt['files']:
        raise ValueError(f'Transfer checksum mismatch: {path}')
    return receipt


def durable_copy(src, dst):
    with Path(src).open('rb') as source, Path(dst).open('wb') as target:
        shutil.copyfileobj(source, target, length=8 * 1024 ** 2)
        target.flush()
        os.fsync(target.fileno())
    return str(dst)


def publish_shard(local, destination):
    """Do not remove the local source until a complete copy passes SHA-256 checks."""
    local, destination = Path(local), Path(destination)
    started = time.perf_counter()
    files = inventory(local)
    if not files or '_SHARD.json' not in files or '_SUCCESS' not in files:
        raise ValueError(f'Cannot publish an unverified/incomplete shard: {local}')
    if destination.exists():
        receipt = verified_receipt(destination)
        if receipt['files'] != files:
            raise FileExistsError(f'Existing destination differs from local shard: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        incoming = destination.with_name(destination.name + '.incoming-' + uuid.uuid4().hex)
        shutil.copytree(local, incoming, copy_function=durable_copy)
        if inventory(incoming) != files:
            raise ValueError(f'Transfer checksum mismatch; local retained: {incoming}')
        receipt = {'verified_at': now(), 'files': files,
                   'stored_bytes': sum(v['bytes'] for v in files.values())}
        atomic_json(incoming / RECEIPT, receipt)
        os.rename(incoming, destination)
    result = json.loads((destination / '_SHARD.json').read_text())
    result.update(run_path=str(destination), stored_bytes=receipt['stored_bytes'],
                  transfer_seconds=time.perf_counter() - started, transferred_at=now())
    shutil.rmtree(local)
    return result


def publish_with_retries(local, destination):
    for attempt in range(3):
        try:
            return publish_shard(local, destination)
        except OSError:
            if attempt == 2:
                raise
            print(f'Transfer retry {attempt + 1}: {destination}', flush=True)
            time.sleep(5)


def shard_ranges(total, size):
    if total < 1 or size < 1:
        raise ValueError('Positive sample count and shard size are required')
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def check_ids(run, expected_ids):
    actual = [sample.sample_id for sample in run]
    if actual != expected_ids or len(set(actual)) != len(actual):
        raise ValueError('Shard sample IDs do not match its ordered dataset slice')


def validate_scope(config):
    if config['models'] != MODELS:
        raise ValueError('Only the three approved MATH generation models are allowed')
    if config['datasets'] != [{'key': 'math', 'task': 'math', 'expected_samples': 5000}]:
        raise ValueError('Only the full 5,000-row MATH dataset is allowed')
    if set(config['model_revisions']) != set(MODELS):
        raise ValueError('Every model needs a pinned revision')


def validate_shard(runner, run, ids):
    check_ids(run, ids)
    checks = verify_run(runner, run)
    width = runner.get_model_spec().hidden_dim
    for sample in run:
        for side in ('pre', 'post'):
            states = sample.get_final_norm_states(side)
            prompt = sample.get_final_norm_states(side, 'prompt_last')
            mean = sample.get_final_norm_states(side, 'mean')
            if states.shape != (sample.n_tokens, width) or prompt.shape != (width,):
                raise ValueError(f'Missing final RMSNorm {side} activations')
            if not np.isfinite(states).all() or not np.isfinite(prompt).all():
                raise ValueError(f'Nonfinite final RMSNorm {side} activations')
            np.testing.assert_allclose(states.mean(0), mean, rtol=1e-5, atol=1e-6)
    return checks


def collect_shard(runner, prepared, items, local, base, generation, capture, identity):
    expected_ids = [item.sample_id for item in items]
    for attempt in range(1, 4):
        if local.exists():
            # Retain incomplete attempts. Accepted runs always live at the canonical path.
            failed = local.with_name(local.name + '.failed-' + uuid.uuid4().hex)
            local.rename(failed)
        try:
            task = SelectedPreparedTask(prepared, items)
            resolved = {**base, 'model': {**base['model'],
                        'identifier': runner.model_name_or_path, 'revision': runner.revision},
                        'collection': {**base['collection'], 'task': 'math',
                        'output': str(local), 'prepared_path': str(prepared)},
                        'math_shard': identity}
            stats = CollectionRunner(runner, task, local, capture_spec=capture,
                                     generation_spec=generation, run_config=resolved).run()
            run = Run(local)
            checks = validate_shard(runner, run, expected_ids)
            summary, seconds = evaluate_run(run, auto_select_evaluator('math'), 'correctness')
            summary['label_path'] = 'labels/correctness.parquet'
            record = {**identity, 'sample_ids': expected_ids, 'samples': len(run),
                      'tokens': stats['n_tokens_total'], 'collect_seconds': stats['duration_seconds'],
                      'eval_seconds': seconds, 'evaluation': summary, 'verification': checks,
                      'finish_reasons': [s.finish_reason for s in run],
                      'created_at': now(), 'attempt': attempt, 'passed': True}
            atomic_json(local / '_SHARD.json', record)
            return
        except Exception:
            print(f'Shard failed (attempt {attempt}): {local}', flush=True)
            traceback.print_exc()
            if attempt == 3:
                raise
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', type=Path, default=root / 'configs/math_full.toml')
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Persistent dami job directory')
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--reserve-gib', type=int, default=64)
    parser.add_argument('--smoke', action='store_true', help='Three MATH rows per model, same 2048-token budget')
    args = parser.parse_args()
    args.local_root = args.local_root.resolve()
    args.output = args.output.resolve()
    if (args.local_root == args.output or args.local_root in args.output.parents
            or args.output in args.local_root.parents):
        parser.error('Local staging and persistent output must be separate directory trees')
    if args.shard_size < 1 or args.reserve_gib < 1:
        parser.error('Shard size and disk reserve must be positive')
    config = tomllib.loads(args.config.read_text())
    validate_scope(config)
    base = load_run_config(str(args.config.parent / config['base_config']))
    # The inherited file contains catalog entries for other experiments, unused here.
    base.pop('model_catalog', None)
    base.pop('safety_judge', None)
    base['collection'].update(task='math')
    generation = GenerationSpec(**{**base['generation'], 'max_new_tokens': config['max_new_tokens']})
    capture = CaptureSpec(**base['capture'])
    if generation.do_sample or generation.temperature != 0 or generation.max_new_tokens != 2048:
        raise ValueError('Expected greedy decoding and the agreed 2048-token cap')
    if (capture.hidden_states_layers is not None or capture.hidden_states_dtype != 'float32'
            or not all((capture.hidden_states, capture.final_norm, capture.save_per_token,
                        capture.save_mean_states, capture.save_prompt_last))):
        raise ValueError('All layers, prompt-last, token, mean, and both norm sides are required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with bf16 is required')
    prepared = Path(config['prepared_root']) / 'math'
    source = PreparedParquetTask(prepared)
    items = list(source.iter_items())
    if (len(items) != 5000 or len({i.sample_id for i in items}) != 5000
            or source.get_prompt_template_metadata()['prompt_template_variant'] != 'zot'):
        raise ValueError('Expected exactly 5000 unique MATH rows prepared with zot')
    selected = items[:3] if args.smoke else items
    source_files = [*sorted(prepared.glob('*.parquet')), prepared / 'prepared_manifest.json']
    code_files = [*sorted((root / 'packages').rglob('*.py')), Path(__file__),
                  root / 'scripts/run_collection_matrix.py', root / 'scripts/check_paper_collection.py']
    plan = {'config': config, 'base': base, 'generation': asdict(generation), 'capture': asdict(capture),
            'mode': 'smoke' if args.smoke else 'full', 'samples_per_model': len(selected),
            'shard_size': args.shard_size, 'models': list(MODELS),
            'data_sha256': {str(p): sha256(p) for p in source_files},
            'code_sha256': {str(p.relative_to(root)): sha256(p) for p in code_files},
            'versions': {name: version(name) for name in ('torch', 'transformers', 'numpy', 'zarr')}}
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    args.local_root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.local_root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = args.output / 'job_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing job has different data/code/config; choose a new output directory')
    else:
        atomic_json(plan_path, {**plan, 'fingerprint': fingerprint, 'created_at': now(),
                              'git_commit': subprocess.check_output(
                                  ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()})
    report = {'fingerprint': fingerprint, 'started_at': now(), 'updated_at': now(),
              'status': 'running', 'pid': os.getpid(), 'local_root': str(args.local_root),
              'output': str(args.output), 'samples_per_model': len(selected),
              'expected_total': len(selected) * len(MODELS), 'rows': [], 'errors': []}
    def save():
        report['updated_at'] = now()
        report['completed_samples'] = sum(row['samples'] for row in report['rows'])
        atomic_json(args.output / 'job_status.json', report)
    def accept(future):
        row = future.result()
        report['rows'].append(row)
        save()
        print(f'PUBLISHED {row["model_alias"]} {row["start"]}:{row["stop"]} '
              f'tokens={row["tokens"]} bytes={row["stored_bytes"]}', flush=True)
    save()
    pending = []
    runner = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for alias, model_id in MODELS.items():
                runner = ModelRunner(model_id, dtype='bfloat16', device_map='cuda:0',
                                     attn_implementation='sdpa', revision=config['model_revisions'][alias])
                try:
                    runner.load()
                    report['current_model'] = alias
                    report['context_preflight'] = context_preflight(runner, items, generation.max_new_tokens)
                    save()
                    for start, stop in shard_ranges(len(selected), args.shard_size):
                        while pending and (pending[0].done() or len(pending) >= 2):
                            accept(pending.pop(0))
                        name = f'shard_{start:05d}_{stop:05d}'
                        local = args.local_root / alias / name
                        destination = args.output / alias / name
                        ids = [i.sample_id for i in selected[start:stop]]
                        identity = {'fingerprint': fingerprint, 'model_alias': alias,
                                    'model_id': model_id, 'start': start, 'stop': stop}
                        if destination.exists():
                            receipt = verified_receipt(destination)
                            row = json.loads((destination / '_SHARD.json').read_text())
                            if any(row[k] != v for k, v in identity.items()) or row['sample_ids'] != ids:
                                raise ValueError(f'Published shard has wrong identity: {destination}')
                            check_ids(Run(destination), ids)
                            row.update(run_path=str(destination), stored_bytes=receipt['stored_bytes'])
                            report['rows'].append(row)
                            if local.exists():
                                # Only delete an identical, successfully transferred local duplicate.
                                publish_shard(local, destination)
                            save()
                            continue
                        if (local / '_SHARD.json').exists():
                            row = json.loads((local / '_SHARD.json').read_text())
                            if any(row[k] != v for k, v in identity.items()):
                                raise ValueError(f'Local shard has wrong identity: {local}')
                            validate_shard(runner, Run(local), ids)
                        else:
                            # Conservative uncompressed upper bound, plus the user's disk reserve.
                            model = runner.get_model_spec()
                            estimated_bytes = int(1.5 * (stop - start) * (generation.max_new_tokens + 3)
                                                  * (runner.probed_n_layers + 2) * model.hidden_dim * 4)
                            needed = args.reserve_gib * GIB + estimated_bytes
                            while shutil.disk_usage(args.local_root).free < needed and pending:
                                accept(pending.pop(0))
                            if shutil.disk_usage(args.local_root).free < needed:
                                raise OSError(f'Insufficient local disk; need {needed / GIB:.1f} GiB free')
                            report['current_shard'] = {'model': alias, 'start': start, 'stop': stop}
                            save()
                            collect_shard(runner, prepared, selected[start:stop], local,
                                          base, generation, capture, identity)
                        pending.append(pool.submit(publish_with_retries, local, destination))
                finally:
                    runner.unload()
                    runner = None
            while pending:
                accept(pending.pop(0))
        for alias in MODELS:
            actual = [sid for row in report['rows'] if row['model_alias'] == alias for sid in row['sample_ids']]
            if actual != [item.sample_id for item in selected]:
                raise ValueError(f'Missing, duplicate, or unordered final MATH coverage for {alias}')
        report['status'] = 'complete'
        report['completed_at'] = now()
        save()
        atomic_json(args.output / '_SUCCESS', {'completed_at': now(), 'samples': report['completed_samples']})
        print(f'COMPLETE: {report["completed_samples"]} MATH generations saved to {args.output}', flush=True)
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append({'at': now(), 'error': str(exc), 'traceback': traceback.format_exc()})
        save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
