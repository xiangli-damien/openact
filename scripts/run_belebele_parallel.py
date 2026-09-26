"""Resume the frozen BELEBELE run with one independent worker per model.

The original collector and its frozen code hashes stay unchanged. This is an
execution-only continuation: identical data, decoding, capture, shard IDs and
verification. A parent holds the legacy job lock; workers own distinct models.
"""
import argparse
import errno
from dataclasses import asdict
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from scripts.run_belebele_collection import (
    ROOT, MODELS, CollectionRunner, ModelRunner, SelectedPreparedTask, Run,
    GenerationSpec, CaptureSpec, load_run_config, tomllib, load_data, validate_scope,
    context_preflight, evaluate_run, auto_select_evaluator, validate_shard,
    check_ids, verified_receipt, publish_with_retries, atomic_json, sha256, now,
    shard_ranges, GIB,
)


def read(path):
    # Status files are atomically replaced on NFS. A reader may briefly see an
    # old handle during the rename; reopen the path rather than failing the job.
    for attempt in range(10):
        try:
            return json.loads(Path(path).read_text())
        except OSError as exc:
            if exc.errno not in (errno.ESTALE, errno.ENOENT) or attempt == 9:
                raise
            time.sleep(.2)


def verify_continuation(plan, cfg, base, generation, capture, inputs):
    for key, actual in [('config', cfg), ('base', base),
                        ('generation', asdict(generation)), ('capture', asdict(capture))]:
        if plan[key] != actual:
            raise ValueError(f'Frozen {key} changed')
    if plan['data_sha256'] != {str(p): sha256(p) for p in inputs}:
        raise ValueError('Frozen source data changed')
    for name, digest in plan['code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen collection code changed: {name}')
    if plan['versions'] != {n: version(n) for n in plan['versions']}:
        raise ValueError('Frozen runtime changed')


def check_coverage(rows, alias, data):
    for lang, (_, items) in data.items():
        actual = [sid for r in rows if r['model_alias'] == alias and r['language'] == lang
                  for sid in r['sample_ids']]
        if actual != [i.sample_id for i in items]:
            raise ValueError(f'Missing, duplicate, or reordered coverage: {alias}/{lang}')


def shared_disk_bound(cfg, dimensions):
    # Reserve both workers' complete worst-case next shard at every admission.
    return sum(int(1.5 * cfg['shard_size'] * (cfg['max_new_tokens'] + 3)
                   * (r['model_layers_including_embedding'] + 2) * r['hidden_dim'] * 4)
               for r in dimensions.values()) + cfg['reserve_gib'] * GIB


def worker(alias, cfg, base, generation, capture, data, plan, execution):
    output, staging = Path(cfg['output_root']), Path(cfg['local_root'])
    phase = output / 'full'
    state_path = phase / alias / 'job_status.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with (state_path.parent / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', phase='full', pid=os.getpid(), current_model=alias,
                     expected_samples=2700, rows=[], preflight={}, started_at=now(),
                     fingerprint=plan['fingerprint'], execution_sha256=execution)

        def save(**updates):
            state.update(updates, updated_at=now())
            for target, source in [('completed_samples', 'samples'),
                                   ('generated_tokens', 'tokens'), ('stored_bytes', 'stored_bytes')]:
                state[target] = sum(r[source] for r in state['rows'])
            atomic_json(state_path, state)

        runner = ModelRunner(cfg['models'][alias], dtype='bfloat16', device_map='cuda:0',
                             attn_implementation='sdpa', revision=cfg['model_revisions'][alias])
        try:
            save(activity='loading')
            runner.load()
            model = runner.get_model_spec()
            record = dict(model=cfg['models'][alias], revision=cfg['model_revisions'][alias],
                          model_generation_defaults=runner.model.generation_config.to_dict(),
                          openact_overrides=asdict(generation),
                          model_layers_including_embedding=runner.probed_n_layers,
                          hidden_dim=model.hidden_dim)
            if record != read(output / 'smoke' / f'{alias}_generation.json'):
                raise ValueError('Resolved generation settings differ from passed smoke')
            target = phase / f'{alias}_generation.json'
            if target.exists() and read(target) != record:
                raise ValueError('Resolved full-run generation settings changed')
            atomic_json(target, record)
            dimensions = {a: read(output / 'smoke' / f'{a}_generation.json') for a in MODELS}
            for lang, (prepared, items) in data.items():
                context = context_preflight(runner, items, generation.max_new_tokens)
                if context['reduced_generation_budget']:
                    raise ValueError('Context would truncate generation budget')
                state['preflight'][f'{alias}/{lang}'] = context
                for start, stop in shard_ranges(len(items), cfg['shard_size']):
                    selected = items[start:stop]
                    ids = [i.sample_id for i in selected]
                    name = f'shard_{start:05d}_{stop:05d}'
                    dest = phase / alias / f'belebele_{lang}' / name
                    local = staging / 'full' / alias / f'belebele_{lang}' / name
                    identity = dict(fingerprint=plan['fingerprint'], phase='full', model_alias=alias,
                                    model_id=cfg['models'][alias], language=lang,
                                    start=start, stop=stop, sample_ids=ids)
                    save(activity='checking', current_language=lang, current_range=[start, stop],
                         current_shard=str(local))
                    if dest.exists():
                        receipt = verified_receipt(dest)
                        row = read(dest / '_SHARD.json')
                        if any(row[k] != v for k, v in identity.items()):
                            raise ValueError('Published shard identity changed')
                        check_ids(Run(dest), ids)
                        row.update(run_path=str(dest), stored_bytes=receipt['stored_bytes'])
                    else:
                        if (local / '_SHARD.json').exists():
                            row = read(local / '_SHARD.json')
                            if any(row[k] != v for k, v in identity.items()):
                                raise ValueError('Staged shard identity changed')
                            validate_shard(runner, Run(local), ids)
                        else:
                            if local.exists():
                                local.rename(local.with_name(local.name + '.incomplete-' + str(time.time_ns())))
                            if shutil.disk_usage(staging).free < shared_disk_bound(cfg, dimensions):
                                raise OSError('SSD cannot reserve both workers plus 50 GiB')
                            local.parent.mkdir(parents=True, exist_ok=True)
                            effective = {**base, 'model': {**base['model'],
                                'identifier': cfg['models'][alias], 'revision': cfg['model_revisions'][alias],
                                'device_map': 'cuda:0'}, 'collection': {**base['collection'],
                                'task': 'belebele', 'language': lang, 'output': str(local),
                                'prepared_path': str(prepared)}}
                            save(activity='collecting')
                            torch.cuda.reset_peak_memory_stats()
                            began = now()
                            stats = CollectionRunner(runner, SelectedPreparedTask(prepared, selected), local,
                                generation_spec=generation, capture_spec=capture, run_config=effective).run()
                            save(activity='activation_audit')
                            run = Run(local)
                            tick = time.perf_counter()
                            checks = validate_shard(runner, run, ids)
                            for sample in run:
                                h = sample.prompt_last_hidden_states
                                if h.shape != (runner.probed_n_layers, model.hidden_dim) or not np.isfinite(h).all():
                                    raise ValueError('Invalid prompt-last hidden state')
                            audit = time.perf_counter() - tick
                            evaluation, seconds = evaluate_run(run, auto_select_evaluator('belebele'), 'correctness')
                            evaluation['label_path'] = 'labels/correctness.parquet'
                            row = dict(**identity, samples=len(run), tokens=stats['n_tokens_total'],
                                collect_seconds=stats['duration_seconds'], audit_seconds=audit,
                                eval_seconds=seconds, evaluation=evaluation, verification=checks,
                                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                finish_reasons=[s.finish_reason for s in run], created_at=now(), passed=True,
                                execution_sha256=execution, collect_started_at=began)
                            atomic_json(local / '_SHARD.json', row)
                        save(activity='publishing')
                        row = publish_with_retries(local, dest)
                    state['rows'].append(row)
                    save()
                    print(json.dumps(dict(model=alias, published=str(dest),
                        completed=state['completed_samples'], collect_seconds=row['collect_seconds'])), flush=True)
            check_coverage(state['rows'], alias, data)
            save(status='complete', activity='complete', completed_at=now())
            atomic_json(phase / alias / '_SUCCESS', dict(samples=2700, fingerprint=plan['fingerprint'],
                        status_sha256=sha256(state_path), completed_at=now()))
        except BaseException:
            save(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            runner.unload()


def supervise(args, cfg, plan, execution):
    output = Path(cfg['output_root'])
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Supervisor interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (output / '.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / '_SUCCESS').exists():
            raise ValueError('Already complete; do not start new workers')
        gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], text=True).split()
        if any(p != str(os.getpid()) for p in gpu_pids):
            raise RuntimeError('Stop the owned legacy collector before the explicit handoff')
        archive = output / f'status_before_parallel_{time.time_ns()}.json'
        atomic_json(archive, read(output / 'job_status.json'))
        children, handles = {}, []
        launch = dict(pid=os.getpid(), started_at=now(), fingerprint=plan['fingerprint'],
                      execution_sha256=execution, command=sys.argv, workers={})
        try:
            for alias in MODELS:
                # Remove no data. New process status replaces stale status only.
                state_path = output / 'full' / alias / 'job_status.json'
                if state_path.exists():
                    atomic_json(state_path.with_name(f'previous_status_{time.time_ns()}.json'), read(state_path))
                atomic_json(state_path, dict(status='starting', rows=[], completed_samples=0))
                handle = (output / f'parallel_{alias}.log').open('a')
                handles.append(handle)
                command = [sys.executable, '-u', '-m', 'scripts.run_belebele_parallel',
                           '--config', str(args.config.resolve()), '--worker', alias]
                children[alias] = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                launch['workers'][alias] = dict(pid=children[alias].pid, command=command)
            atomic_json(output / 'parallel_launch.json', launch)
            while True:
                states = {a: read(output / 'full' / a / 'job_status.json') for a in MODELS}
                rows = [r for s in states.values() for r in s.get('rows', [])]
                exit_codes = {a: p.poll() for a, p in children.items()}
                finished = all(v is not None for v in exit_codes.values())
                failed = any(v not in (None, 0) for v in exit_codes.values())
                complete = finished and not failed and all(s['status'] == 'complete' for s in states.values())
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True).strip()
                status = dict(status='complete' if complete else ('failed' if finished else
                              ('running_with_failure' if failed else 'running')),
                    phase='full', mode='parallel', pid=os.getpid(), started_at=launch['started_at'],
                    updated_at=now(), fingerprint=plan['fingerprint'], execution_sha256=execution,
                    expected_samples=5400, completed_samples=sum(r['samples'] for r in rows),
                    generated_tokens=sum(r['tokens'] for r in rows), stored_bytes=sum(r['stored_bytes'] for r in rows),
                    workers={a: {k: v for k, v in s.items() if k != 'rows'} for a, s in states.items()},
                    exit_codes=exit_codes, gpu_csv=gpu, rows=rows)
                atomic_json(output / 'job_status.json', status)
                atomic_json(output / 'full' / 'job_status.json', status)
                with (output / 'parallel_resources.jsonl').open('a') as h:
                    h.write(json.dumps(dict(at=now(), gpu_csv=gpu,
                        completed={a:s.get('completed_samples',0) for a,s in states.items()}))+'\n')
                if finished:
                    if not complete or status['completed_samples'] != 5400:
                        raise RuntimeError('Parallel job did not complete; inspect per-model logs')
                    for root in (output / 'full', output):
                        atomic_json(root / '_SUCCESS', dict(samples=5400, completed_at=now(),
                            fingerprint=plan['fingerprint'], execution_sha256=execution,
                            status_sha256=sha256(output/'job_status.json')))
                    break
                time.sleep(10)
        except BaseException:
            atomic_json(output / 'parallel_supervisor_error.json', dict(at=now(), traceback=traceback.format_exc()))
            raise
        finally:
            for p in children.values():
                if p.poll() is None:
                    p.terminate()
            for p in children.values():
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            for h in handles:
                h.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/belebele_full.toml')
    p.add_argument('--worker', choices=list(MODELS))
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    base = load_run_config(str(args.config.parent / cfg['base_config']))
    base.pop('model_catalog', None); base.pop('safety_judge', None)
    base['collection'].update(task='belebele', template='zot')
    generation = GenerationSpec(**{**base['generation'], 'max_new_tokens':cfg['max_new_tokens']})
    capture = CaptureSpec(**base['capture'])
    validate_scope(cfg, generation, capture)
    data, inputs = load_data(cfg)
    output = Path(cfg['output_root'])
    plan = read(output/'job_plan.json')
    verify_continuation(plan, cfg, base, generation, capture, inputs)
    smoke = read(output/'smoke/job_status.json')
    if smoke['status'] != 'complete' or smoke['completed_samples'] != 12 or not (output/'smoke/_SUCCESS').exists():
        raise ValueError('All six smoke cells must already be verified')
    execution = sha256(Path(__file__))
    provenance = dict(execution_sha256=execution, parent_fingerprint=plan['fingerprint'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_at=now(), policy='User-authorized two-model concurrency; original collection contract unchanged')
    receipt = output / 'executions' / f'{execution}.json'
    if not receipt.exists():
        atomic_json(receipt, provenance)
    if args.prepare_only:
        print(json.dumps(provenance)); return
    torch.set_num_threads(4)
    if args.worker:
        worker(args.worker, cfg, base, generation, capture, data, plan, execution)
    else:
        supervise(args, cfg, plan, execution)


if __name__ == '__main__':
    main()
