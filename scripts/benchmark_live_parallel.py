"""Benchmark a large MMLU worker against the live MATH collector.

A shadow MATH baseline uses exactly the next production questions. The paired
case reuses the live MATH model, so only two large models need be resident.
The existing small MMLU worker is paused (its extra memory is reported separately).
An independent watchdog resumes both production processes after controller loss.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_collection_concurrency import write_json, alive, resume, gpu_sample, worker


def read(path):
    return json.loads(Path(path).read_text())


def progress(log):
    with log.open('rb') as handle:
        handle.seek(max(0, log.stat().st_size - 32768))
        text = handle.read().decode(errors='replace')
    bars = re.findall(r'Collecting:[^\r\n]*?OK=(\d+)[^\r\n]*', text)
    return int(bars[-1]) if bars else 0


def process_memory():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(p): int(m) for p, m in (line.split(',') for line in text.splitlines())}


def guard(args):
    deadline = time.time() + 1500
    while time.time() < deadline and alive(args.controller):
        if (args.output / '_RESUMED').exists():
            return
        time.sleep(2)
    # Owned benchmark workers only; stop their GPU work before restoring production.
    registry = args.output / 'workers.json'
    for pid in read(registry) if registry.exists() else []:
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'benchmark_live_parallel' in cmd and str(args.output).encode() in cmd:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    for pid in (args.math_pid, args.mmlu_pid):
        resume(pid)
    write_json(args.output / '_WATCHDOG_RESUMED.json', {'at': time.time()})


def select_window(done, shard_size, samples):
    # Exclude the request that may already be in flight when SIGSTOP arrives.
    start, stop = done + 1, done + 1 + samples
    if start < 1 or stop >= shard_size:
        raise ValueError('Not enough untouched rows left in the current shard')
    return start, stop


def same_tokens(a, b):
    return a['sample_ids'] == b['sample_ids'] and a['token_ids'] == b['token_ids']


def assessment(serial_math, serial_mmlu, paired_math, paired_mmlu, peak, paused, total=40960):
    serial = serial_math['collect_seconds'] + serial_mmlu['collect_seconds']
    parallel = max(paired_math['collection_finished'], paired_mmlu['collection_finished']) - min(
        paired_math['started'], paired_mmlu['started'])
    result = {'status': 'complete', 'tokens_identical': same_tokens(serial_math, paired_math)
              and same_tokens(serial_mmlu, paired_mmlu),
              'activations_valid': bool(paired_math['passed'] and paired_mmlu['passed']),
              'serial_collection_seconds': serial, 'parallel_collection_seconds': parallel,
              'combined_speedup': serial / parallel,
              'observed_peak_gpu_mib': peak, 'paused_extra_mmlu_mib': paused,
              'projected_peak_gpu_mib': peak - paused,
              'projected_headroom_mib': total - peak + paused}
    result['eligible'] = (result['tokens_identical'] and result['activations_valid']
                          and result['combined_speedup'] >= 1.05
                          and result['projected_headroom_mib'] >= 8192)
    return result


def live_tokens_and_checks(path, start, stop, ids):
    import numpy as np
    import zarr
    group = zarr.open_group(str(path / 'tensors.zarr'), mode='r')
    ptr = group['tokens/sample_ptr'][:]
    tokens = []
    for row in range(start, stop):
        a, b = int(ptr[row]), int(ptr[row + 1])
        if a < 0 or b <= a:
            raise ValueError('Production token row is not complete')
        tokens.append(group['tokens/ids'][a:b].tolist())
        hs = group['hidden_states/per_token'][a:b]
        prompt = group['hidden_states/prompt_last'][row]
        if not np.isfinite(hs).all() or not np.isfinite(prompt).all():
            raise ValueError('Nonfinite production hidden states')
        np.testing.assert_allclose(hs.mean(0), group['hidden_states/mean'][row], rtol=1e-5, atol=1e-6)
        for side in ('pre', 'post'):
            norm = group[f'final_norm/{side}/per_token'][a:b]
            if norm.shape != (b-a, hs.shape[-1]) or not np.isfinite(norm).all():
                raise ValueError('Invalid production RMSNorm activations')
            if not np.isfinite(group[f'final_norm/{side}/prompt_last'][row]).all():
                raise ValueError('Invalid prompt-last RMSNorm activations')
            np.testing.assert_allclose(norm.mean(0), group[f'final_norm/{side}/mean'][row], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(hs[:, -1], group['final_norm/post/per_token'][a:b])
    return {'sample_ids': ids, 'token_ids': tokens, 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--math-root', type=Path)
    parser.add_argument('--mmlu-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['qwen2', 'llama3', 'llama32'])
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--dataset', choices=['math', 'mmlu'])
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--barrier', type=Path)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--controller', type=int)
    parser.add_argument('--math-pid', type=int)
    parser.add_argument('--mmlu-pid', type=int)
    args = parser.parse_args()
    if args.guard:
        return guard(args)
    if args.worker:
        if args.dataset == 'math':
            from scripts import run_collection_matrix as matrix
            matrix.select_items = lambda items, *unused: items[args.start:args.stop]
        return worker(args)
    if not 1 <= args.samples <= 12:
        parser.error('Use between 1 and 12 requests per workload')
    math, mmlu = read(args.math_root / 'job_status.json'), read(args.mmlu_root / 'job_status.json')
    args.math_pid, args.mmlu_pid = math['pid'], mmlu['pid']
    for job, path in ((math, args.math_root), (mmlu, args.mmlu_root)):
        cmd = Path(f'/proc/{job["pid"]}/cmdline').read_bytes()
        if job['status'] != 'running' or str(path).encode() not in cmd:
            raise RuntimeError('Expected a live authorized collection process')
    memories = process_memory()
    paused_extra = memories.get(args.mmlu_pid, 0)
    if paused_extra > 4608:
        raise RuntimeError('Wait for MMLU to unload; cannot fit three resident large models')
    log = ROOT / 'logs' / f'{args.math_root.name}.log'
    done = progress(log)
    shard = math['current_shard']
    start, stop = select_window(done, shard['stop']-shard['start'], args.samples)
    local = Path(math['local_root']) / math['current_model'] / f'shard_{shard["start"]:05d}_{shard["stop"]:05d}'
    if not (local / 'tensors.zarr').exists() or (local / '_SHARD.json').exists():
        raise RuntimeError('MATH is not currently inside a collectable shard')
    args.output.mkdir(parents=True, exist_ok=False)
    processes, handles, resources = [], [], []
    deadline = time.monotonic() + 1380
    report = {'status': 'running', 'math_model': math['current_model'], 'mmlu_model': args.model,
              'math_fingerprint': read(args.math_root / 'job_plan.json')['fingerprint'],
              'mmlu_fingerprint': read(args.mmlu_root / 'job_plan.json')['fingerprint'],
              'math_shard': shard, 'production_local_range': [start, stop],
              'samples_per_workload': args.samples, 'started_at': time.time(),
              'method': 'Identical token workloads; live MATH, shadow MMLU; collection time excludes evaluation',
              'timing_resolution_seconds': .05, 'cases': {}}
    write_json(args.output / 'benchmark.json', report)
    watchdog = subprocess.Popen([sys.executable, '-m', 'scripts.benchmark_live_parallel', '--guard',
                  '--controller', str(os.getpid()), '--math-pid', str(args.math_pid),
                  '--mmlu-pid', str(args.mmlu_pid), '--output', str(args.output)], cwd=ROOT,
                  start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    last_sample = 0
    def tick(case):
        nonlocal last_sample
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded benchmark deadline exceeded')
        if time.monotonic() - last_sample > .4:
            sample = {**gpu_sample(), 'case': case, 'processes': process_memory()}
            resources.append(sample)
            last_sample = time.monotonic()
            if sample['memory_mib'] > 37888:
                raise RuntimeError('Benchmark exceeded the physical 3 GiB emergency reserve')
        if not alive(args.math_pid) or not alive(args.mmlu_pid):
            raise RuntimeError('Production process disappeared during benchmark')
    def launch(name, model, dataset):
        path = args.output / name
        barrier = args.output / f'{name}.start'
        handle = (args.output / f'{name}.log').open('w')
        handles.append(handle)
        cmd = [sys.executable, '-m', 'scripts.benchmark_live_parallel', '--worker', '--model', model,
               '--dataset', dataset, '--samples', str(args.samples), '--output', str(path),
               '--barrier', str(barrier)]
        if dataset == 'math':
            cmd += ['--start', str(shard['start']+start), '--stop', str(shard['start']+stop)]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        processes.append(proc)
        write_json(args.output / 'workers.json', [p.pid for p in processes])
        while not (path / 'ready.json').exists():
            tick(name)
            if proc.poll() is not None:
                raise RuntimeError(f'Worker failed loading: {name}')
            time.sleep(.1)
        return proc, path, barrier
    def finish(job, name):
        proc, path, _ = job
        while proc.poll() is None:
            tick(name)
            time.sleep(.1)
        if proc.returncode:
            raise RuntimeError(f'Worker failed: {name}; inspect its log')
        return read(path / 'result.json')
    try:
        lease = {'controller_pid': os.getpid(), 'expires_at': time.time()+1500,
                 'reason': 'Controlled parallel benchmark; independent resume watchdog active',
                 'report': str(args.output / 'benchmark.json')}
        for path in (args.math_root, args.mmlu_root):
            write_json(path / 'maintenance_lease.json', lease)
        for pid in (args.math_pid, args.mmlu_pid):
            os.kill(pid, signal.SIGSTOP)
        time.sleep(1)
        # Re-read after stopping, so a just-finished row cannot shift the comparison.
        done = progress(log)
        start, stop = select_window(done, shard['stop']-shard['start'], args.samples)
        report['production_local_range'] = [start, stop]
        for name, model, dataset in [('solo_math', math['current_model'], 'math'),
                                     ('solo_mmlu', args.model, 'mmlu')]:
            print(f'BENCHMARK {name} {model}', flush=True)
            job = launch(name, model, dataset)
            job[2].touch()
            report['cases'][name] = finish(job, name)
            write_json(args.output / 'benchmark.json', report)
        print(f'BENCHMARK paired live {math["current_model"]}/math + {args.model}/mmlu', flush=True)
        job = launch('paired_mmlu', args.model, 'mmlu')
        resume(args.math_pid)
        while progress(log) < start:
            tick('paired_warmup')
            time.sleep(.05)
        paired_start = time.monotonic()
        job[2].touch()
        while progress(log) < stop:
            tick('paired')
            if job[0].poll() not in (None, 0):
                raise RuntimeError('Paired MMLU failed')
            time.sleep(.05)
        paired_end = time.monotonic()
        os.kill(args.math_pid, signal.SIGSTOP)
        report['cases']['paired_mmlu'] = finish(job, 'paired')
        paired_math = live_tokens_and_checks(local, start, stop, report['cases']['solo_math']['sample_ids'])
        paired_math.update(started=paired_start, collection_finished=paired_end,
                           collect_seconds=paired_end-paired_start)
        report['cases']['paired_math'] = paired_math
        peak = max(s['memory_mib'] for s in resources if s['case'].startswith('paired'))
        report.update(assessment(report['cases']['solo_math'], report['cases']['solo_mmlu'],
                                 paired_math, report['cases']['paired_mmlu'], peak, paused_extra))
        report['mmlu_peak_process_mib'] = max(s['processes'].get(job[0].pid, 0) for s in resources)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc(), eligible=False)
        raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for pid in (args.math_pid, args.mmlu_pid):
            resume(pid)
        for handle in handles:
            handle.close()
        report['finished_at'] = time.time()
        report['production_resumed'] = True
        write_json(args.output / 'benchmark.json', report)
        write_json(args.output / 'resources.json', resources)
        (args.output / '_RESUMED').touch()
        for path in (args.math_root, args.mmlu_root):
            (path / 'maintenance_lease.json').unlink(missing_ok=True)
        watchdog.wait(timeout=10)
    if args.policy and report.get('eligible'):
        import hashlib
        policy = read(args.policy) if args.policy.exists() else {'pairs': {}}
        policy['pairs'][f'{math["current_model"]}:{args.model}'] = {
            'report': str(args.output / 'benchmark.json'),
            'sha256': hashlib.sha256((args.output / 'benchmark.json').read_bytes()).hexdigest()}
        write_json(args.policy, policy)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases', 'error')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
