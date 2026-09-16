"""Compare identical serial/parallel MATH + MMLU workloads on one GPU.

The production MATH process is paused, never terminated, during the controlled
comparison. An independent watchdog resumes it if this controller exits/crashes.
Benchmark outputs are separate from production datasets.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def resume(pid):
    if alive(pid):
        os.kill(pid, signal.SIGCONT)


def watchdog(args):
    # Outlives the controller, including SIGKILL; bounded pause even if it hangs.
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline and alive(args.controller_pid):
        if (args.output / '_RESUMED').exists():
            return
        time.sleep(2)
    resume(args.math_pid)
    write_json(args.output / '_WATCHDOG_RESUMED.json', {'time': time.time()})


def gpu_sample():
    values = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True).splitlines()[0].split(',')
    return {'time': time.time(), 'memory_mib': int(values[0]), 'utilization': int(values[1])}


def worker(args):
    import torch
    from openact_collect import CollectionRunner, ModelRunner
    from openact_collect.config import load_run_config, tomllib
    from openact_collect.schema import CaptureSpec, GenerationSpec
    from openact_collect.tasks.prepared import PreparedParquetTask
    from openact_core import Run
    from openact_eval.evaluators.registry import auto_select_evaluator
    try:
        from scripts.run_collection_matrix import SelectedPreparedTask, select_items, evaluate_run
        from scripts.run_math_collection import validate_shard
    except ModuleNotFoundError:
        from run_collection_matrix import SelectedPreparedTask, select_items, evaluate_run
        from run_math_collection import validate_shard
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / 'configs/math_full.toml').read_text())
    base = load_run_config(str(root / 'configs/capability.toml'))
    prepared = Path(config['prepared_root']) / args.dataset
    items = select_items(list(PreparedParquetTask(prepared).iter_items()), args.samples, 42, args.dataset)
    runner = ModelRunner(config['models'][args.model], dtype='bfloat16', device_map='cuda:0',
                         attn_implementation='sdpa', revision=config['model_revisions'][args.model])
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        runner.load()
        write_json(args.output / 'ready.json', {'pid': os.getpid()})
        while not args.barrier.exists():
            time.sleep(.05)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        stats = CollectionRunner(runner, SelectedPreparedTask(prepared, items), args.output / 'run',
                                 capture_spec=CaptureSpec(**base['capture']),
                                 generation_spec=GenerationSpec(**base['generation']),
                                 run_config={'benchmark': True, 'model': args.model,
                                             'dataset': args.dataset, 'settings': base}).run()
        collection_finished = time.monotonic()
        run = Run(args.output / 'run')
        verification = validate_shard(runner, run, [item.sample_id for item in items])
        evaluation, _ = evaluate_run(run, auto_select_evaluator(args.dataset), 'correctness')
        report = {'model': args.model, 'dataset': args.dataset, 'samples': len(items),
                  'started': started, 'collection_finished': collection_finished,
                  'finished': time.monotonic(), 'collect_seconds': stats['duration_seconds'],
                  'tokens': stats['n_tokens_total'],
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'sample_ids': [s.sample_id for s in run],
                  'token_ids': [s.token_ids.tolist() for s in run],
                  'verification': verification, 'evaluation': evaluation, 'passed': True}
        write_json(args.output / 'result.json', report)
    finally:
        runner.unload()


def run_case(args, name, jobs, resources):
    case = args.output / name
    case.mkdir()
    barrier = case / 'start'
    processes = []
    handles = []
    started = time.monotonic()
    try:
        for model, dataset in jobs:
            handle = (case / f'{model}_{dataset}.log').open('w')
            handles.append(handle)
            path = case / f'{model}_{dataset}'
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                   '--model', model, '--dataset', dataset, '--samples', str(args.samples),
                   '--output', str(path), '--barrier', str(barrier)]
            processes.append((subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT), path))
        while not all((p / 'ready.json').exists() for _, p in processes):
            if any(proc.poll() is not None for proc, _ in processes):
                raise RuntimeError(f'Worker failed loading: {case}')
            if time.monotonic() - started > 180:
                raise TimeoutError(f'Workers did not become ready: {case}')
            resources.append({'case': name, **gpu_sample()})
            time.sleep(.5)
        barrier.write_text(str(time.monotonic()))
        while any(proc.poll() is None for proc, _ in processes):
            resources.append({'case': name, **gpu_sample()})
            if time.monotonic() - started > 1200:
                raise TimeoutError(f'Benchmark case timeout: {case}')
            time.sleep(.5)
        if any(proc.returncode for proc, _ in processes):
            raise RuntimeError(f'Benchmark worker failed: {case}')
        results = [json.loads((p / 'result.json').read_text()) for _, p in processes]
        wall = max(r['finished'] for r in results) - min(r['started'] for r in results)
        collection_wall = max(r['collection_finished'] for r in results) - min(r['started'] for r in results)
        return {'wall_seconds': wall, 'collection_wall_seconds': collection_wall, 'workers': results,
                'peak_total_gpu_mib': max(r['memory_mib'] for r in resources if r['case'] == name)}
    finally:
        for proc, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for handle in handles:
            handle.close()


def assess(solo_math, solo_mmlu, pair, min_speedup=1.05):
    serial = solo_math['wall_seconds'] + solo_mmlu['wall_seconds']
    by_task = {w['dataset']: w for w in pair['workers']}
    matching = all(
        by_task[task]['sample_ids'] == solo['workers'][0]['sample_ids']
        and by_task[task]['token_ids'] == solo['workers'][0]['token_ids']
        for task, solo in [('math', solo_math), ('mmlu', solo_mmlu)]
    )
    speedup = serial / pair['wall_seconds']
    return {'serial_seconds': serial, 'parallel_seconds': pair['wall_seconds'],
            'combined_speedup': speedup, 'tokens_identical': matching,
            'peak_total_gpu_mib_including_paused_math': pair['peak_total_gpu_mib'],
            'memory_headroom_mib': 40960 - pair['peak_total_gpu_mib'],
            'eligible': matching and speedup >= min_speedup and pair['peak_total_gpu_mib'] <= 32768}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--math-pid', type=int)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume-watchdog', action='store_true')
    parser.add_argument('--controller-pid', type=int)
    parser.add_argument('--model', choices=['llama32', 'qwen2', 'llama3'])
    parser.add_argument('--dataset', choices=['math', 'mmlu'])
    parser.add_argument('--barrier', type=Path)
    args = parser.parse_args()
    if args.resume_watchdog:
        return watchdog(args)
    if args.worker:
        return worker(args)
    if not args.math_pid or not alive(args.math_pid):
        parser.error('A live production MATH pid is required')
    args.output.mkdir(parents=True, exist_ok=False)
    resources = []
    report = {'status': 'running', 'math_pid': args.math_pid, 'samples_per_worker': args.samples,
              'started_at': time.time(), 'cases': {}, 'comparisons': {}}
    write_json(args.output / 'benchmark.json', report)
    guard = subprocess.Popen([
        sys.executable, str(Path(__file__).resolve()), '--resume-watchdog',
        '--math-pid', str(args.math_pid), '--controller-pid', str(os.getpid()),
        '--output', str(args.output),
    ], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        os.kill(args.math_pid, signal.SIGSTOP)
        # Allow the current CUDA call to finish; the model stays resident.
        time.sleep(2)
        report['cases']['solo_mmlu'] = run_case(args, 'solo_mmlu', [('llama32', 'mmlu')], resources)
        write_json(args.output / 'benchmark.json', report)
        for alias in ['llama32', 'qwen2', 'llama3']:
            print(f'BASELINE {alias}/math', flush=True)
            solo = run_case(args, f'solo_math_{alias}', [(alias, 'math')], resources)
            report['cases'][f'solo_math_{alias}'] = solo
            print(f'PARALLEL {alias}/math + llama32/mmlu', flush=True)
            pair = run_case(args, f'parallel_{alias}', [(alias, 'math'), ('llama32', 'mmlu')], resources)
            report['cases'][f'parallel_{alias}'] = pair
            comparison = assess(solo, report['cases']['solo_mmlu'], pair)
            report['comparisons'][alias] = comparison
            print(json.dumps({'primary_model': alias, **comparison}), flush=True)
            write_json(args.output / 'benchmark.json', report)
        report['status'] = 'complete'
        report['all_pairs_eligible'] = all(r['eligible'] for r in report['comparisons'].values())
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        resume(args.math_pid)
        (args.output / '_RESUMED').touch()
        report['finished_at'] = time.time()
        report['production_math_resumed'] = True
        write_json(args.output / 'benchmark.json', report)
        write_json(args.output / 'gpu_samples.json', resources)
        guard.wait(timeout=10)


if __name__ == '__main__':
    main()
