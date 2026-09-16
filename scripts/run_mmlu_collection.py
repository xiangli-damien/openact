"""Collect all three full MMLU datasets while preserving the existing MATH job.

The 1B MMLU worker may share the GPU only for benchmark-approved combinations.
The two large MMLU models wait until the entire MATH job has completed. Uses
the proven shard format/copy checks without changing the running MATH code.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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

import torch
from openact_collect import CollectionRunner, ModelRunner
from openact_collect.config import load_run_config, tomllib
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_core import Run
from openact_eval.evaluators.registry import auto_select_evaluator

try:
    from scripts.run_math_collection import (
        MODELS, GIB, atomic_json, now, sha256, shard_ranges, check_ids,
        validate_shard, verified_receipt, publish_shard, publish_with_retries,
    )
    from scripts.run_collection_matrix import SelectedPreparedTask, context_preflight, evaluate_run
except ModuleNotFoundError:
    from run_math_collection import (
        MODELS, GIB, atomic_json, now, sha256, shard_ranges, check_ids,
        validate_shard, verified_receipt, publish_shard, publish_with_retries,
    )
    from run_collection_matrix import SelectedPreparedTask, context_preflight, evaluate_run


def validate_scope(config):
    if config['models'] != MODELS or set(config['model_revisions']) != set(MODELS):
        raise ValueError('Expected the three approved, pinned MMLU models')
    if config['datasets'] != [{'key': 'mmlu', 'task': 'mmlu', 'expected_samples': 14042}]:
        raise ValueError('Expected full 14,042-row MMLU only')


def validate_benchmark(benchmark):
    if benchmark.get('status') != 'complete' or set(benchmark.get('comparisons', {})) != set(MODELS):
        raise ValueError('A completed comparison for all three MATH models is required')
    for result in benchmark['comparisons'].values():
        if not result['tokens_identical'] or result['memory_headroom_mib'] < 8192:
            raise ValueError('All pairings must pass correctness and memory headroom checks')
    if not any(r['eligible'] for r in benchmark['comparisons'].values()):
        raise ValueError('No combination showed a useful throughput improvement')


def may_run(alias, primary_complete, primary_status, comparisons):
    if primary_complete:
        return True
    if alias != 'llama32' or primary_status.get('status') != 'running':
        return False
    comparison = comparisons.get(primary_status.get('current_model'), {})
    return bool(comparison.get('eligible'))


class MathPriorityGate:
    def __init__(self, primary, benchmark, alias, callback):
        self.primary = primary
        self.comparisons = benchmark['comparisons']
        self.alias = alias
        self.callback = callback

    def wait(self, runner):
        waiting = False
        while True:
            try:
                state = json.loads((self.primary / 'job_status.json').read_text())
                complete = (self.primary / '_SUCCESS').exists() and state['status'] == 'complete'
                permitted = may_run(self.alias, complete, state, self.comparisons)
            except (OSError, ValueError, KeyError):
                permitted = False
            if permitted:
                if not runner._loaded:
                    runner.load()
                if waiting:
                    self.callback('collecting', None)
                return
            if runner._loaded:
                runner.unload()
            if not waiting:
                self.callback('waiting_for_math', self.alias)
                waiting = True
            time.sleep(10)


class GatedCollectionRunner(CollectionRunner):
    def __init__(self, *args, gate, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate

    def _process_safe(self, item):
        # Recheck at sample boundaries so the small worker yields on model changes.
        self.gate.wait(self.model_manager)
        return super()._process_safe(item)


def collect_shard(runner, gate, prepared, items, local, base, generation, capture, identity):
    ids = [item.sample_id for item in items]
    for attempt in range(1, 4):
        if local.exists():
            local.rename(local.with_name(local.name + '.failed-' + uuid.uuid4().hex))
        try:
            gate.wait(runner)
            effective = {**base,
                         'model': {**base['model'], 'identifier': runner.model_name_or_path,
                                   'revision': runner.revision, 'device_map': 'cuda:0'},
                         'collection': {**base['collection'], 'task': 'mmlu',
                                        'prepared_path': str(prepared), 'output': str(local)},
                         'mmlu_shard': identity}
            torch.cuda.reset_peak_memory_stats()
            stats = GatedCollectionRunner(
                runner, SelectedPreparedTask(prepared, items), local, gate=gate,
                capture_spec=capture, generation_spec=generation, run_config=effective,
            ).run()
            gate.wait(runner)
            run = Run(local)
            verification = validate_shard(runner, run, ids)
            summary, seconds = evaluate_run(run, auto_select_evaluator('mmlu'), 'correctness')
            summary['label_path'] = 'labels/correctness.parquet'
            atomic_json(local / '_SHARD.json', {
                **identity, 'sample_ids': ids, 'samples': len(run), 'tokens': stats['n_tokens_total'],
                'collect_seconds': stats['duration_seconds'], 'eval_seconds': seconds,
                'evaluation': summary, 'verification': verification,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'finish_reasons': [s.finish_reason for s in run],
                'created_at': now(), 'attempt': attempt, 'passed': True,
            })
            return
        except Exception:
            traceback.print_exc()
            if attempt == 3:
                raise
            torch.cuda.empty_cache()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'configs/mmlu_full.toml')
    parser.add_argument('--primary-math', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--reserve-gib', type=int, default=160,
                        help='Leave extra local disk capacity for the existing MATH writer')
    args = parser.parse_args()
    for field in ('local_root', 'output', 'primary_math', 'benchmark'):
        setattr(args, field, getattr(args, field).resolve())
    if (args.local_root == args.output or args.local_root in args.output.parents
            or args.output in args.local_root.parents):
        parser.error('Local staging and dami must be separate directory trees')
    if args.shard_size < 1 or args.reserve_gib < 160:
        parser.error('Positive shards and at least 160 GiB shared-disk reserve are required')
    config = tomllib.loads(args.config.read_text())
    validate_scope(config)
    benchmark = json.loads(args.benchmark.read_text())
    validate_benchmark(benchmark)
    base = load_run_config(str(args.config.parent / config['base_config']))
    base.pop('model_catalog', None)
    base.pop('safety_judge', None)
    base['collection'].update(task='mmlu')
    generation = GenerationSpec(**{**base['generation'], 'max_new_tokens': config['max_new_tokens']})
    capture = CaptureSpec(**base['capture'])
    if generation.do_sample or generation.temperature != 0 or generation.max_new_tokens != 2048:
        raise ValueError('Expected greedy decoding and the agreed 2048-token cap')
    if (capture.hidden_states_layers is not None or capture.hidden_states_dtype != 'float32'
            or not all((capture.hidden_states, capture.final_norm, capture.save_per_token,
                        capture.save_mean_states, capture.save_prompt_last))):
        raise ValueError('All raw layers, prompt-last, tokens, means, and RMS sides are required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with bf16 is required')
    prepared = Path(config['prepared_root']) / 'mmlu'
    source = PreparedParquetTask(prepared)
    items = list(source.iter_items())
    subjects = Counter(item.meta.get('subject') for item in items)
    if (len(items) != 14042 or len({i.sample_id for i in items}) != 14042
            or source.get_prompt_template_metadata()['prompt_template_variant'] != 'zot'):
        raise ValueError('Expected 14,042 unique MMLU rows prepared with zot')
    if len(subjects) != 57 or set(subjects) != {s['config'] for s in source.dataset_sources}:
        raise ValueError('Expected every one of the 57 pinned MMLU subjects')
    source_files = [*sorted(prepared.glob('*.parquet')), prepared / 'prepared_manifest.json']
    code_files = [*sorted((root / 'packages').rglob('*.py')), Path(__file__),
                  root / 'scripts/run_math_collection.py', root / 'scripts/run_collection_matrix.py',
                  root / 'scripts/check_paper_collection.py']
    plan = {'config': config, 'base': base, 'generation': asdict(generation), 'capture': asdict(capture),
            'samples_per_model': len(items), 'shard_size': args.shard_size, 'models': list(MODELS),
            'subject_counts': dict(subjects),
            'data_sha256': {str(p): sha256(p) for p in source_files},
            'code_sha256': {str(p.relative_to(root)): sha256(p) for p in code_files},
            'benchmark_sha256': sha256(args.benchmark), 'primary_math': str(args.primary_math),
            'policy': '1B shares approved pairings; large MMLU models wait for MATH success',
            'versions': {n: version(n) for n in ('torch', 'transformers', 'numpy', 'zarr')}}
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    args.local_root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.local_root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = args.output / 'job_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing MMLU job has different data/code/config')
    else:
        atomic_json(plan_path, {**plan, 'fingerprint': fingerprint, 'created_at': now(),
                              'git_commit': subprocess.check_output(
                                  ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()})
    report = {'fingerprint': fingerprint, 'started_at': now(), 'status': 'running', 'pid': os.getpid(),
              'local_root': str(args.local_root), 'output': str(args.output),
              'samples_per_model': len(items), 'expected_total': len(items) * len(MODELS),
              'rows': [], 'errors': []}
    def save():
        report['updated_at'] = now()
        report['completed_samples'] = sum(r['samples'] for r in report['rows'])
        atomic_json(args.output / 'job_status.json', report)
    def stage(name, detail):
        report['stage'] = name
        report['wait_detail'] = detail
        save()
        print(f'STAGE {name}: {detail}', flush=True)
    def accept(future):
        row = future.result()
        report['rows'].append(row)
        save()
        print(f'PUBLISHED {row["model_alias"]} {row["start"]}:{row["stop"]} '
              f'tokens={row["tokens"]} bytes={row["stored_bytes"]}', flush=True)
    save()
    pending = []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for alias, model_id in MODELS.items():
                # Finish previous transfers before waiting for the next GPU admission.
                while pending:
                    accept(pending.pop(0))
                runner = ModelRunner(model_id, dtype='bfloat16', device_map='cuda:0',
                                     attn_implementation='sdpa', revision=config['model_revisions'][alias])
                gate = MathPriorityGate(args.primary_math, benchmark, alias, stage)
                report['current_model'] = alias
                save()
                try:
                    gate.wait(runner)
                    report['context_preflight'] = context_preflight(runner, items, generation.max_new_tokens)
                    save()
                    for start, stop in shard_ranges(len(items), args.shard_size):
                        while pending and (pending[0].done() or len(pending) >= 2):
                            accept(pending.pop(0))
                        name = f'shard_{start:05d}_{stop:05d}'
                        local = args.local_root / alias / name
                        destination = args.output / alias / name
                        ids = [i.sample_id for i in items[start:stop]]
                        identity = {'fingerprint': fingerprint, 'model_alias': alias, 'model_id': model_id,
                                    'dataset': 'mmlu', 'start': start, 'stop': stop}
                        if destination.exists():
                            receipt = verified_receipt(destination)
                            row = json.loads((destination / '_SHARD.json').read_text())
                            if any(row[k] != v for k, v in identity.items()) or row['sample_ids'] != ids:
                                raise ValueError('Published MMLU shard has wrong identity')
                            check_ids(Run(destination), ids)
                            row.update(run_path=str(destination), stored_bytes=receipt['stored_bytes'])
                            report['rows'].append(row)
                            if local.exists():
                                publish_shard(local, destination)
                            save()
                            continue
                        gate.wait(runner)
                        if (local / '_SHARD.json').exists():
                            row = json.loads((local / '_SHARD.json').read_text())
                            if any(row[k] != v for k, v in identity.items()):
                                raise ValueError('Local MMLU shard has wrong identity')
                            validate_shard(runner, Run(local), ids)
                        else:
                            model = runner.get_model_spec()
                            needed = args.reserve_gib * GIB + int(1.5 * (stop - start)
                                     * (generation.max_new_tokens + 3)
                                     * (runner.probed_n_layers + 2) * model.hidden_dim * 4)
                            while shutil.disk_usage(args.local_root).free < needed:
                                if pending:
                                    accept(pending.pop(0))
                                else:
                                    runner.unload()
                                    stage('waiting_for_disk', {'required_free_gib': needed / GIB})
                                    time.sleep(30)
                            gate.wait(runner)
                            report['current_shard'] = {'model': alias, 'start': start, 'stop': stop}
                            stage('collecting', None)
                            collect_shard(runner, gate, prepared, items[start:stop], local,
                                          base, generation, capture, identity)
                        pending.append(pool.submit(publish_with_retries, local, destination))
                finally:
                    runner.unload()
            while pending:
                accept(pending.pop(0))
        for alias in MODELS:
            actual = [sid for r in report['rows'] if r['model_alias'] == alias for sid in r['sample_ids']]
            if actual != [i.sample_id for i in items]:
                raise ValueError(f'Missing, duplicate, or unordered MMLU coverage: {alias}')
        report.update(status='complete', completed_at=now())
        save()
        atomic_json(args.output / '_SUCCESS', {'completed_at': now(), 'samples': report['completed_samples']})
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append({'at': now(), 'error': str(exc), 'traceback': traceback.format_exc()})
        save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
