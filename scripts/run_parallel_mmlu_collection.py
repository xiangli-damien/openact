"""Keep the original collection identity; replace only GPU admission scheduling.

The legacy launcher is deliberately unchanged: its source and all scientific
settings are part of existing shard fingerprints. This adapter records separate
execution provenance and admits large models using an auditable live benchmark.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_mmlu_collection as legacy
from scripts.run_math_collection import atomic_json, now, sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False


def active_lease(path, at=None):
    try:
        lease = read_json(path)
        return (lease if lease['expires_at'] > (time.time() if at is None else at)
                and process_alive(lease['controller_pid']) else None)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pair_admitted(result):
    """Never accept a bare eligible flag; require the recorded measurements."""
    return bool(result.get('status') == 'complete' and result.get('tokens_identical')
                and result.get('activations_valid')
                and result.get('combined_speedup', 0) >= 1.05
                and result.get('projected_headroom_mib', 0) >= 8192)


def gpu_memory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    return tuple(map(int, out.splitlines()[0].split(',')))


def own_gpu_memory():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    for line in out.splitlines():
        pid, memory = map(int, line.split(','))
        if pid == os.getpid():
            return memory
    return 0


class ParallelGate:
    policy_path = None
    output = None

    def __init__(self, primary, benchmark, alias, callback):
        self.primary, self.benchmark, self.alias, self.callback = primary, benchmark, alias, callback
        self.cached = {}

    def evidence(self, primary_alias):
        try:
            policy = read_json(self.policy_path)
            entry = policy['pairs'][f'{primary_alias}:{self.alias}']
            key = (entry['report'], entry['sha256'])
            if key not in self.cached:
                path = Path(entry['report'])
                if sha256(path) != entry['sha256']:
                    return None
                record = read_json(path)
                if (record['math_model'] != primary_alias or record['mmlu_model'] != self.alias
                        or record['math_fingerprint'] != read_json(self.primary / 'job_plan.json')['fingerprint']
                        or record['mmlu_fingerprint'] != read_json(self.output / 'job_plan.json')['fingerprint']):
                    return None
                self.cached[key] = record
            return self.cached[key]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def wait(self, runner):
        last_stage = None
        while True:
            reason = 'waiting_for_parallel_admission'
            permitted = False
            try:
                state = read_json(self.primary / 'job_status.json')
                complete = (self.primary / '_SUCCESS').exists() and state['status'] == 'complete'
                evidence = self.evidence(state.get('current_model'))
                permitted = (legacy.may_run(self.alias, complete, state, self.benchmark['comparisons'])
                             or (state['status'] == 'running' and pair_admitted(evidence or {})))
                if active_lease(self.output / 'scheduling_pause.json'):
                    reason, permitted = 'waiting_for_benchmark', False
                if permitted:
                    used, total = gpu_memory()
                    # A loaded worker yields before the next request if reserve is lost.
                    # Before loading, account for its measured peak process allocation.
                    # The measured process peak includes its CUDA context. That
                    # context may already be counted in total device usage.
                    need = (0 if runner._loaded else max(0,
                            (evidence or {}).get('mmlu_peak_process_mib', 4096) - own_gpu_memory()))
                    if total - used - need < 8192:
                        reason, permitted = 'waiting_for_gpu_memory', False
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                permitted = False
            if permitted:
                if not runner._loaded:
                    runner.load()
                if last_stage is not None:
                    self.callback('collecting', None)
                return
            if runner._loaded:
                runner.unload()
            if last_stage != reason:
                self.callback(reason, {'model': self.alias, 'policy': str(self.policy_path)})
                last_stage = reason
            time.sleep(5)


def checked_receipts(output, workers=4):
    """Full SHA-256 verification, parallelized across immutable published shards."""
    paths = sorted(p.parent for p in output.glob('*/shard_*/_COPY_VERIFIED.json')
                   if '.incoming-' not in str(p) and '.failed-' not in str(p))
    result = {}
    original = legacy.verified_receipt
    def verify(path):
        receipt = original(path)
        return path, receipt, sha256(path / '_COPY_VERIFIED.json')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, receipt, digest in pool.map(verify, paths):
            result[path] = receipt, digest
            print(f'RESUME VERIFIED {path.parent.name}/{path.name}', flush=True)
    def cached(path):
        path = Path(path)
        previous = result.pop(path, None)
        if previous and sha256(path / '_COPY_VERIFIED.json') == previous[1]:
            return previous[0]
        return original(path)
    return cached


def handoff_ready(status):
    return (status.get('status') == 'running'
            and status.get('current_model') in ('qwen2', 'llama3')
            and status.get('stage') == 'waiting_for_math')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--after-legacy', action='store_true',
                        help='Wait for the existing 1B worker to finish before replacing its idle launcher')
    opts, remaining = parser.parse_known_args()
    def argument(name):
        return Path(remaining[remaining.index(name) + 1]).resolve()
    output = argument('--output')
    policy = opts.policy.resolve()
    scheduler_lock = (output / '.parallel_scheduler.lock').open('a')
    fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if opts.after_legacy:
        print('Waiting for legacy 1B MMLU completion; both production jobs remain running.', flush=True)
        while True:
            status = read_json(output / 'job_status.json')
            if status['status'] == 'complete':
                return
            if not process_alive(status['pid']) or status['status'] == 'failed':
                raise RuntimeError('Existing MMLU failed; inspect it before changing the scheduler')
            if handoff_ready(status):
                cmd = Path(f'/proc/{status["pid"]}/cmdline').read_bytes()
                if b'scripts/run_mmlu_collection.py' not in cmd or str(output).encode() not in cmd:
                    raise RuntimeError('Refusing to signal a different collector')
                print('1B complete; switching the idle MMLU launcher to measured parallel admission.', flush=True)
                os.kill(status['pid'], signal.SIGINT)
                deadline = time.monotonic() + 120
                while process_alive(status['pid']):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Legacy MMLU did not exit; not starting a duplicate')
                    time.sleep(1)
                break
            time.sleep(15)
    # Additional scheduling provenance is explicit; historical job_plan and shards stay intact.
    record = {'at': now(), 'pid': os.getpid(), 'adapter_sha256': sha256(Path(__file__)),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'policy_path': str(policy), 'legacy_capture_fingerprint': read_json(output / 'job_plan.json')['fingerprint'],
              'policy': 'Approved exact model pairs may share MATH; require >=1.05x measured throughput and 8 GiB headroom',
              'resume_validation': 'Full per-file SHA-256, four published shards at a time'}
    atomic_json(output / 'scheduler_execution.json', record)
    previous = read_json(output / 'job_status.json')
    atomic_json(output / 'job_status_before_scheduler.json', previous)
    atomic_json(output / 'job_status.json', {
        **previous, 'pid': os.getpid(), 'status': 'running', 'errors': [],
        'stage': 'waiting_for_resume_verification', 'updated_at': now(),
    })
    legacy.verified_receipt = checked_receipts(output)
    ParallelGate.policy_path, ParallelGate.output = policy, output
    legacy.MathPriorityGate = ParallelGate
    sys.argv = [str(ROOT / 'scripts/run_mmlu_collection.py'), *remaining]
    legacy.main()


if __name__ == '__main__':
    main()
