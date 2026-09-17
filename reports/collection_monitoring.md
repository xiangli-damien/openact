# Collection monitoring

The read-only `scripts/monitor_collections.py` samples the existing MATH and MMLU
jobs every 30 seconds on gpu2. It does not load models or modify collection data.
It runs in tmux session `openact-collection-monitor`, independently of SSH.

Output: `/lambda/nfs/dami/openact/runs/collection_monitor_20260916/`

- `status.json`: latest resource snapshot, process state, published counts per
  model, current progress bar, pending transfers, active alerts.
- `history.jsonl`: resource/progress history. GPU peaks are **sampled** peaks;
  brief spikes between samples can be missed.
- `events.jsonl`: new log errors, alert transitions/recoveries, model completion.
- `state.json`: durable log offsets and progress/alert state for monitor restarts.

Published counts come from shard records with copy receipts and completion
markers, since launcher status can lag completed background copies. This is
metadata monitoring, not a repeated full checksum of activation files. Original
publish-time SHA-256 verification and final coverage checks remain authoritative.

Alerts: stopped/missing/failed processes; new CUDA OOM/tracebacks/retries; no
collection progress for 10 minutes; pending transfer without a published shard
for 10 minutes (diagnostic warning, not proof of failure); GPU memory >=90% for
60 seconds (critical >=95%); MemAvailable <16 GiB (critical <8); local disk <96 GiB
(critical <64); inconsistent published metadata or completion. A long legitimate
transfer may trigger a warning and needs inspection. `waiting_for_math` is an
expected scheduling state. `waiting_for_disk` gets its own warning. System memory
checks use MemAvailable, not MemFree, so reclaimable file cache is not misreported.

The Codex thread heartbeat **math-mmlu**, named “监控 MATH/MMLU 采集与内存”,
checks every 30 minutes and reviews new events, diagnoses anomalies, and reports
new failures, recovery, model completion, or required user action. Normal and
unchanged states stay quiet. The heartbeat needs the local computer and Codex
desktop app running; remote collection and telemetry continue independently.
It pauses after final MATH 15,000 / MMLU 42,126 coverage, labels and transfer checks.

Launch on gpu2 from `/lambda/nfs/dami/openact`:

```sh
tmux new-session -d -s openact-collection-monitor \
  'cd /lambda/nfs/dami/openact && exec .venv/bin/python -u scripts/monitor_collections.py --output /lambda/nfs/dami/openact/runs/collection_monitor_20260916 >> logs/collection_monitor_20260916.log 2>&1'
```

The monitor holds an exclusive lock to reject duplicate instances. It stops when
both jobs have successful top-level markers and complete published counts; the
heartbeat subsequently performs the final data audit. It never kills or restarts
collection workers automatically, nor changes precision, token caps, or datasets.
