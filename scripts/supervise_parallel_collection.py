"""Measure newly active MATH pairings and hand off MMLU after its 1B run.

No changes to model/data/capture settings. Unqualified combinations stay queued.
Run in tmux; exclusive lock prevents two admission controllers on one job.
"""
import argparse
from contextlib import contextmanager
import os
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_live_parallel import progress, process_memory, select_window, read
from scripts.benchmark_collection_concurrency import write_json


def alive_handoff(pid, output):
    try:
        root = Path('/proc') / str(pid)
        cmd = (root / 'cmdline').read_bytes()
        return (b'run_parallel_mmlu_collection' in cmd and str(output).encode() in cmd
                and '\nState:\tZ' not in (root / 'status').read_text())
    except OSError:
        return False


@contextmanager
def admission_pause(root):
    """The upgraded collector unloads between requests before a shadow is loaded."""
    lease = root / 'scheduling_pause.json'
    owned = False
    try:
        status = read(root / 'job_status.json')
        cmd = Path(f'/proc/{status["pid"]}/cmdline').read_bytes()
        if b'run_parallel_mmlu_collection' in cmd:
            write_json(lease, {'controller_pid': os.getpid(), 'expires_at': time.time()+1500,
                               'reason': 'Release the production MMLU model for a bounded pairing benchmark'})
            owned = True
            deadline = time.monotonic() + 300
            while True:
                current = read(root / 'job_status.json')
                if (current.get('stage') == 'waiting_for_benchmark'
                        and process_memory().get(current['pid'], 0) <= 4608):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('MMLU did not reach an unloaded request boundary')
                time.sleep(1)
        yield
    finally:
        if owned:
            lease.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--math-root', type=Path, required=True)
    parser.add_argument('--mmlu-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = args.output / 'admission.json'
    if not policy.exists():
        write_json(policy, {'pairs': {}})
    status_file = args.output / 'supervisor.json'
    state = read(status_file) if status_file.exists() else {'attempts': {}}
    state.update(status='running', pid=__import__('os').getpid())
    write_json(status_file, state)
    mmlu = read(args.mmlu_root / 'job_status.json')
    command = [sys.executable, '-u', '-m', 'scripts.run_parallel_mmlu_collection',
               '--after-legacy', '--policy', str(policy), '--primary-math', str(args.math_root),
               '--benchmark', str(ROOT / 'reports/concurrency_benchmark.json'),
               '--local-root', mmlu['local_root'], '--output', str(args.mmlu_root)]
    # This waiter owns the subsequent MMLU process. It does not start a second collector.
    if not alive_handoff(state.get('handoff_pid'), args.mmlu_root):
        with (ROOT / 'logs' / f'{args.mmlu_root.name}.log').open('a') as log:
            handoff = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        state['handoff_pid'] = handoff.pid
    write_json(status_file, state)
    while True:
        math, mmlu = read(args.math_root / 'job_status.json'), read(args.mmlu_root / 'job_status.json')
        if not alive_handoff(state['handoff_pid'], args.mmlu_root) and mmlu['status'] != 'complete':
            raise RuntimeError('MMLU scheduler handoff failed; inspect the collection log')
        if math['status'] == 'complete' and (args.math_root / '_SUCCESS').exists():
            state.update(status='math_complete', updated_at=time.time())
            write_json(status_file, state)
            return
        if math['status'] != 'running' or mmlu['status'] not in ('running', 'complete'):
            time.sleep(15)
            continue
        if mmlu.get('stage') == 'waiting_for_resume_verification':
            time.sleep(15)
            continue
        alias = math['current_model']
        if alias not in ('qwen2', 'llama3'):
            time.sleep(15)
            continue
        for candidate in ('qwen2', 'llama3'):
            key = f'{alias}:{candidate}'
            if key in read(policy)['pairs'] or key in state['attempts']:
                continue
            with admission_pause(args.mmlu_root):
                try:
                    math, mmlu = read(args.math_root / 'job_status.json'), read(args.mmlu_root / 'job_status.json')
                    if math['current_model'] != alias or mmlu['status'] != 'running':
                        break
                    if process_memory().get(mmlu['pid'], 0) > 4608:
                        break
                    shard = math['current_shard']
                    select_window(progress(ROOT / 'logs' / f'{args.math_root.name}.log'),
                                  shard['stop']-shard['start'], 6)
                except (OSError, ValueError, KeyError):
                    break
                output = args.output / f'{alias}_{candidate}_{int(time.time())}'
                command = [sys.executable, '-u', '-m', 'scripts.benchmark_live_parallel',
                           '--math-root', str(args.math_root), '--mmlu-root', str(args.mmlu_root),
                           '--output', str(output), '--model', candidate, '--samples', '6',
                           '--policy', str(policy)]
                print(f'Measuring {key}: {output}', flush=True)
                with (args.output / f'{output.name}.log').open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                record = read(output / 'benchmark.json') if (output / 'benchmark.json').exists() else {}
                state['attempts'][key] = {'returncode': result.returncode, 'report': str(output / 'benchmark.json'),
                                          'eligible': record.get('eligible', False),
                                          'error': record.get('error'), 'at': time.time()}
                write_json(status_file, state)
                print(json.dumps({key: state['attempts'][key]}), flush=True)
        time.sleep(15)


if __name__ == '__main__':
    main()
