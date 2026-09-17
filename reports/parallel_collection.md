# MATH / MMLU parallel admission

The current HF collectors retain their exact models, revisions, full datasets,
zero-shot CoT, greedy decoding, 2048-token cap, teacher-forced extraction, and
float32 storage of all hidden layers, prompt-last states, generation tokens and
both sides of final RMSNorm. Data still stages locally and publishes to dami
only after copy verification. No vLLM conversion is involved.

`supervise_parallel_collection.py` keeps the current MATH and 1B MMLU processes
running. It measures newly encountered large-model pairs and arranges a handoff
after the legacy 1B MMLU run completes. `run_parallel_mmlu_collection.py` then
replaces only GPU admission, without changing the source files included in the
existing scientific job fingerprint. The separate `scheduler_execution.json`
records the new scheduling code, commit and policy; historical job plans remain
unchanged. Resume performs full per-file SHA-256 verification with four shards
in flight, then reuses those verified receipts once during launcher replay.

## Admission evidence

`benchmark_live_parallel.py` compares the same questions and generated token IDs
in serial and paired execution. It uses a shadow MATH worker for the serial case
and the live production MATH worker for the paired case, with separate temporary
MMLU outputs. The in-flight MATH question is excluded from timing. MATH is paused
at the next completed-request boundary when its finite benchmark workload ends.
Timing compares collection only, excluding evaluation, and includes saving the
full activations. The paired production MATH tensors are checked for finite
values, means, prompt-last fields and final-norm correspondence. Normal shard
validation and evaluation still run later before publication.

Acceptance requires identical tokens, valid activations, at least 1.05x combined
throughput, and at least 8 GiB projected production GPU headroom. If the old 1B
MMLU model is still resident but paused, its measured extra memory is explicitly
subtracted from the physical peak to estimate the future two-worker pairing.
The benchmark aborts above 37,888 MiB physical usage on this 40 GiB GPU. Results
are short pilots, not proof that every long input has the same peak: the new
launcher rechecks free GPU memory at every request boundary and yields when the
8 GiB reserve is unavailable. A process cannot interrupt a CUDA operation that
has already started; existing shard retry and failure monitoring remain active.

Every deliberate pause has a 25-minute independent resume watchdog, a bounded
monitoring lease, and finally-block restoration. Benchmark outputs are isolated
from production coverage. Failed or unmeasured combinations remain queued; the
1B pairings retain their original measured approvals. The supervisor tries each
large pairing once per active MATH model and records failures for inspection.
An SSH disconnect does not stop it. It does not relaunch failed collectors blindly.

## Launch (gpu2)

From `/lambda/nfs/dami/openact`, after GitHub push and GPU fast-forward pull:

```sh
.venv/bin/python -u -m scripts.supervise_parallel_collection \
  --math-root /lambda/nfs/dami/openact/runs/math_full_20260916 \
  --mmlu-root /lambda/nfs/dami/openact/runs/mmlu_full_20260916 \
  --output /lambda/nfs/dami/openact/runs/parallel_admission_20260917
```

Run in tmux `openact-parallel-admission`. `supervisor.json` gives attempts and the
handoff PID; `admission.json` points to SHA-256-pinned benchmark reports. Production
progress remains in the existing collection monitor. The user's Codex heartbeat
checks every **30 minutes** and reports meaningful changes or failures.
