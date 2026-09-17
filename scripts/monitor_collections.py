"""Read-only Linux telemetry for the authorized MATH/MMLU collection jobs.

Runs independently of SSH/Codex. It records events; the thread heartbeat delivers
notifications and diagnoses failures. Sampling cannot capture every brief peak.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


GIB = 1024 ** 3
MODELS = ('llama32', 'qwen2', 'llama3')
SHARD = re.compile(r'shard_\d{5}_\d{5}$')
PROGRESS = re.compile(r'Collecting:[^\r\n]*?(\d+)/(\d+)[^\r\n]*')
ERROR = re.compile(r'CUDA.*out of memory|OutOfMemoryError|Traceback \(most recent call last\)|'
                   r'Shard failed|Transfer retry|Err=[1-9]\d*', re.I)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def maintenance(root, now):
    """Honor only a bounded lease whose controller is still alive."""
    try:
        lease = read_json(root / 'maintenance_lease.json')
        if lease['expires_at'] <= now or lease['expires_at'] - now > 1800:
            return None
        pid = int(lease['controller_pid'])
        info = (Path('/proc') / str(pid) / 'status').read_text()
        if re.search(r'^State:\s+[ZX]', info, re.M):
            return None
        return lease
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_json(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def meminfo(text):
    values = {k: int(v.split()[0]) * 1024 for k, v in
              (line.split(':', 1) for line in text.splitlines())}
    return {k: values[k] for k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}


def process(pid, job_path):
    root = Path('/proc') / str(pid)
    try:
        info = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines())
        cmd = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        state = info['State'].strip().split()[0]
        result = {'pid': pid, 'state': state, 'matches_job': str(job_path) in cmd,
                  'rss_bytes': int(info.get('VmRSS', '0 kB').split()[0]) * 1024}
        result['alive'] = result['matches_job'] and state not in ('Z', 'X')
        result['wchar'] = int(dict(line.split(':', 1) for line in
                                   (root / 'io').read_text().splitlines()).get('wchar', 0))
        return result
    except FileNotFoundError:
        return {'pid': pid, 'alive': False, 'state': 'missing', 'wchar': 0}


def log_snapshot(path, state):
    stat = path.stat()
    with path.open('rb') as handle:
        handle.seek(max(0, stat.st_size - 65536))
        tail = handle.read().decode(errors='replace')
        # On first inspection include recent errors, not the entire historical log.
        offset = state.get('offset', max(0, stat.st_size - 65536))
        if state.get('inode', stat.st_ino) != stat.st_ino or offset > stat.st_size:
            offset = 0
        handle.seek(offset)
        new = handle.read(1024 * 1024).decode(errors='replace')
        state.update(offset=handle.tell(), inode=stat.st_ino)
    matches = list(PROGRESS.finditer(tail))
    progress = None
    if matches:
        last = matches[-1]
        ok = re.search(r'OK=(\d+)', last.group())
        err = re.search(r'Err=(\d+)', last.group())
        progress = {'done': int(ok[1]) if ok else int(last[1]), 'total': int(last[2]),
                    'errors': int(err[1]) if err else 0}
    errors = list(dict.fromkeys(line.strip()[:500] for line in re.split(r'[\r\n]', new)
                               if ERROR.search(line)))
    return {'bytes': stat.st_size, 'modified_at_epoch': stat.st_mtime,
            'current_bar': progress}, errors


class PublishedShards:
    """Cache small summaries; never rescan/hash activation files on each poll."""
    def __init__(self):
        self.cache = {}

    def scan(self, root, fingerprint):
        totals = {m: {'samples': 0, 'tokens': 0, 'bytes': 0, 'shards': 0} for m in MODELS}
        accepted, problems = set(), []
        for model in MODELS:
            parent = root / model
            if not parent.exists():
                continue
            for path in parent.iterdir():
                if not SHARD.fullmatch(path.name):
                    continue
                try:
                    receipt_path, record_path = path / '_COPY_VERIFIED.json', path / '_SHARD.json'
                    key = (str(path), receipt_path.stat().st_mtime_ns, record_path.stat().st_mtime_ns)
                    if key not in self.cache:
                        receipt, record = read_json(receipt_path), read_json(record_path)
                        if (not record['passed'] or record['fingerprint'] != fingerprint
                                or record['model_alias'] != model
                                or record['samples'] != record['stop'] - record['start']
                                or not (path / '_SUCCESS').is_file()
                                or 'labels/correctness.parquet' not in receipt['files']
                                or record['evaluation']['n_error'] != 0
                                or record['evaluation']['n_evaluated'] != record['samples']):
                            raise ValueError('Incomplete or inconsistent published shard metadata')
                        self.cache[key] = {'samples': record['samples'], 'tokens': record['tokens'],
                                           'bytes': receipt['stored_bytes'], 'shards': 1}
                    for k, value in self.cache[key].items():
                        totals[model][k] += value
                    accepted.add(str(path))
                except (OSError, ValueError, KeyError) as exc:
                    problems.append(f'{path}: {exc}')
        return totals, accepted, problems


def advance(state, marker, now, waiting=False):
    if waiting or state.get('marker') != marker:
        state.update(marker=marker, last_progress=now)
    return max(0, now - state.setdefault('last_progress', now))


def pressure(resources, state, now):
    issues = {}
    gpu = resources['gpu']
    fraction = gpu['used_mib'] / gpu['total_mib']
    if fraction >= .9:
        since = state.setdefault('gpu_pressure_since', now)
        if now - since >= 60:
            issues['gpu_memory'] = 'critical' if fraction >= .95 else 'warning'
    else:
        state.pop('gpu_pressure_since', None)
    available = resources['memory']['MemAvailable'] / GIB
    if available < 16:
        issues['system_memory'] = 'critical' if available < 8 else 'warning'
    free = resources['disk_free_bytes'] / GIB
    if free < 96:
        issues['local_disk'] = 'critical' if free < 64 else 'warning'
    return issues


def transitions(previous, current):
    return ([{'kind': 'alert', 'key': k, 'severity': v} for k, v in current.items()
             if previous.get(k) != v] +
            [{'kind': 'recovery', 'key': k} for k in previous if k not in current])


def sample_job(root, logs, cache, state, now):
    status = read_json(root / 'job_status.json')
    published, accepted, problems = cache.scan(root, status['fingerprint'])
    total = sum(v['samples'] for v in published.values())
    log, errors = log_snapshot(logs / (root.name + '.log'), state.setdefault('log', {}))
    proc = process(status['pid'], root)
    pending = []
    for record in Path(status['local_root']).glob('*/shard_*/_SHARD.json'):
        if SHARD.fullmatch(record.parent.name):
            destination = root / record.parent.parent.name / record.parent.name
            if str(destination) not in accepted:
                pending.append(str(record.parent))
    stage = status.get('stage', 'collecting')
    complete = (status['status'] == 'complete' and (root / '_SUCCESS').exists()
                and total == status['expected_total'] and not problems)
    marker = [total, status.get('current_model'), status.get('current_shard'), log['current_bar']]
    # During transfer, actual process writes count as progress; status can lag copies.
    if pending:
        marker.append(proc.get('wchar', 0))
    lease = maintenance(root, now)
    idle = advance(state, marker, now, waiting=stage.startswith('waiting_') or complete or bool(lease))
    transfer_idle = advance(state.setdefault('transfer', {}), total, now, waiting=not pending)
    issues = {}
    if not complete:
        if not proc['alive']:
            issues['process_missing'] = 'critical'
        elif proc['state'] in ('T', 't') and not lease:
            issues['process_stopped'] = 'warning'
        if status['status'] == 'failed' or status['errors']:
            issues['job_failed'] = 'critical'
        if idle >= 600:
            issues['no_progress_10min'] = 'warning'
        if stage == 'waiting_for_disk':
            issues['waiting_for_disk'] = 'warning'
        if pending and transfer_idle >= 600 and not lease:
            issues['no_published_transfer_10min'] = 'warning'
    if problems:
        issues['published_metadata'] = 'critical'
    if status['status'] == 'complete' and not complete:
        issues['completion_inconsistent'] = 'critical'
    result = {'status': status['status'], 'stage': stage, 'complete': complete,
              'current_model': status.get('current_model'), 'current_shard': status.get('current_shard'),
              'expected_total': status['expected_total'], 'published_samples': total,
              'published': published, 'process': proc, 'log': log,
              'pending_transfer_shards': pending, 'idle_seconds': idle,
              'transfer_idle_seconds': transfer_idle,
              'metadata_problems': problems, 'job_errors': status['errors'],
              'maintenance': lease}
    return result, issues, errors


def resources(disk):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    used, total, utilization = map(int, output.splitlines()[0].split(','))
    return {'gpu': {'used_mib': used, 'total_mib': total, 'utilization_percent': utilization},
            'memory': meminfo(Path('/proc/meminfo').read_text()),
            'disk_free_bytes': shutil.disk_usage(disk).free}


def parallel_scheduler(root, mmlu_complete):
    paths = sorted((root / 'runs').glob('parallel_admission_*/supervisor.json'))
    if not paths:
        return None, {}
    path = paths[-1]
    status = read_json(path)
    supervisor = process(status['pid'], path.parent)
    handoff = process(status['handoff_pid'], root / 'runs/mmlu_full_20260916')
    issues = {}
    if status['status'] == 'running' and not supervisor['alive']:
        issues['parallel/supervisor_missing'] = 'warning'
    if not mmlu_complete and not handoff['alive']:
        issues['parallel/handoff_missing'] = 'critical'
    if any(v['returncode'] != 0 for v in status.get('attempts', {}).values()):
        issues['parallel/benchmark_failed'] = 'warning'
    return {'path': str(path), 'status': status['status'], 'supervisor': supervisor,
            'handoff': handoff, 'attempts': status.get('attempts', {})}, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/lambda/nfs/dami/openact'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('Use at least five seconds between samples')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.output / 'state.json'
    state = read_json(state_path) if state_path.exists() else {}
    cache = PublishedShards()
    while True:
        started = time.time()
        snapshot = {'at': utc(), 'epoch': started, 'monitor_pid': os.getpid(),
                    'interval_seconds': args.interval, 'jobs': {}}
        issues, events = {}, []
        try:
            snapshot['resources'] = resources('/home/ubuntu')
            gpu_used = snapshot['resources']['gpu']['used_mib']
            state['sampled_peak_gpu_mib'] = max(state.get('sampled_peak_gpu_mib', 0), gpu_used)
            snapshot['sampled_peak_gpu_mib'] = state['sampled_peak_gpu_mib']
            issues.update(pressure(snapshot['resources'], state, started))
        except Exception as exc:
            issues['resource_probe'] = 'warning'
            snapshot['resource_probe_error'] = str(exc)
        for dataset in ('math', 'mmlu'):
            try:
                path = args.root / 'runs' / (dataset + '_full_20260916')
                job, alerts, errors = sample_job(path, args.root / 'logs', cache,
                                                state.setdefault(dataset, {}), started)
                snapshot['jobs'][dataset] = job
                issues.update({f'{dataset}/{k}': v for k, v in alerts.items()})
                events.extend({'kind': 'log_error', 'job': dataset, 'detail': e} for e in errors)
                completed = state.setdefault('completed_models', [])
                for model, summary in job['published'].items():
                    key = dataset + '/' + model
                    if summary['samples'] == job['expected_total'] // 3 and key not in completed:
                        events.append({'kind': 'model_complete', 'key': key, 'samples': summary['samples']})
                        completed.append(key)
            except Exception as exc:
                issues[dataset + '/probe_failed'] = 'warning'
                snapshot['jobs'][dataset] = {'probe_error': str(exc), 'complete': False}
        try:
            snapshot['parallel_scheduler'], alerts = parallel_scheduler(
                args.root, snapshot['jobs'].get('mmlu', {}).get('complete', False))
            issues.update(alerts)
        except Exception as exc:
            issues['parallel/probe_failed'] = 'warning'
            snapshot['parallel_scheduler_error'] = str(exc)
        events.extend(transitions(state.get('active_alerts', {}), issues))
        state['active_alerts'] = snapshot['active_alerts'] = issues
        done = all(j['complete'] for j in snapshot['jobs'].values())
        if done:
            events.append({'kind': 'all_jobs_complete'})
        for event in events:
            append_json(args.output / 'events.jsonl', {'at': snapshot['at'], **event})
        append_json(args.output / 'history.jsonl', snapshot)
        save_json(args.output / 'status.json', snapshot)
        save_json(state_path, state)
        print(json.dumps({'at': snapshot['at'], 'alerts': issues, 'events': len(events)}), flush=True)
        if args.once or done:
            break
        time.sleep(max(1, args.interval - (time.time() - started)))


if __name__ == '__main__':
    main()
