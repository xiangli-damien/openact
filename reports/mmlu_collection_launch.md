# Full MMLU collection beside the existing MATH job

## Authorized scope

2026-09-16: benchmark throughput and GPU memory first; if viable, start all three
models on full MMLU. The selected models remain Llama-3.2-1B-Instruct,
Qwen2-7B-Instruct, and Meta-Llama-3-8B-Instruct. Each gets all **14,042 test rows
across 57 subjects**, totaling **42,126 generations**. `configs/mmlu_full.toml`
pins model revisions; the prepared `cais/mmlu` source revision is
`c30699e8356da336a370243923dbaf21066bb9fe`.

HF bf16, greedy zero-shot CoT, max 2,048 new tokens, a separate teacher-forced
forward over the exact prompt and response IDs, all hidden layers, final RMSNorm
input/output, prompt-last, every generated token, and generation means. Raw
activations are stored as float32. Each shard gets MMLU correctness labels.

## Controlled concurrency benchmark

`scripts/benchmark_collection_concurrency.py` uses the same 12 deterministically
selected questions per task for serial and parallel cases. It measures each
MATH model alone, the 1B MMLU worker alone, and each MATH model paired with the
1B MMLU worker. All cases run the production capture/validation/evaluation path
on local disk, with the unchanged 2,048-token generation cap.

The production MATH process is paused in memory during the comparison and
resumed in `finally`. An independent watchdog also resumes it after controller
failure or a 40-minute deadline. No production shard is discarded or regenerated.
Its in-progress shard's wall time includes this pause; exclude that interruption
when using its duration to estimate normal collection speed.

Workers finish model loading before a shared start barrier. The measured wall
time includes generation, teacher-forced extraction, local writes, activation
checks, and evaluation; it excludes checkpoint loading and NFS transfers. The
parallel duration is the interval from the first worker's start to the last
worker's finish. The speedup compares that duration against the sum of the two
corresponding standalone durations. Different questions are never used to claim
a concurrency speedup.

Acceptance requires identical generated token IDs and ordered sample IDs,
passing activation/evaluation checks, at least 5% measured combined throughput
gain, and at least 8 GiB GPU memory headroom. GPU usage is sampled approximately
once per second, including model loading; worker CUDA allocation/reservation
peaks are also recorded. Total GPU memory measurements include the extra paused
production 1B model, so production pairs do not need that extra resident copy.

This is a small admission benchmark, not a confidence interval or a guarantee
that every long-running shard will see the same gain. A concurrent individual
MATH job may slow down even when the combined workload finishes faster.

## Queue and storage policy

`scripts/run_mmlu_collection.py` does not modify or restart the existing MATH
launcher or package code. It checks the benchmark and the MATH status at sample
boundaries:

- 1B MMLU can run beside each benchmark-approved MATH model.
- If the current pairing is unapproved, 1B MMLU unloads and waits, preserving
  already written rows in its current shard.
- Qwen2 and Llama-3 MMLU wait for the entire MATH job's `_SUCCESS` before loading.
  This prevents two large models from competing for the 40 GB card and preserves
  the original MATH queue's priority.

The MMLU job uses local 100-row shards, a single background copier, at most two
pending transfers, and a 160 GiB local free-space reserve plus a conservative
size allowance for the next shard. The larger reserve leaves room for the
independent MATH writer. Transfer uses the existing fsync/SHA-256/atomic-publish
helper; local duplicates are deleted only after verification. Each completed
shard is independently readable as an OpenAct `Run`.

Restarting the same command validates data/code/config fingerprints and skips
verified published shards. Partial failed shards are retained for diagnosis and
restarted as separate attempts. All 42,126 ordered IDs must pass final coverage
checks before the job creates its top-level `_SUCCESS`.

## Runtime paths on gpu2

- Local staging: `/home/ubuntu/openact-staging/mmlu_full_20260916`
- Persistent output: `/lambda/nfs/dami/openact/runs/mmlu_full_20260916`
- Log: `/lambda/nfs/dami/openact/logs/mmlu_full_20260916.log`
- MATH dependency: `/lambda/nfs/dami/openact/runs/math_full_20260916`
- Benchmark artifacts: `/home/ubuntu/openact-staging/concurrency_benchmark_20260916`
- Benchmark log: `/lambda/nfs/dami/openact/logs/concurrency_benchmark_20260916.log`

Launch from `/lambda/nfs/dami/openact`, in detached tmux `openact-mmlu-full`:

```bash
.venv/bin/python -u scripts/run_mmlu_collection.py \
  --primary-math /lambda/nfs/dami/openact/runs/math_full_20260916 \
  --benchmark /lambda/nfs/dami/openact/reports/concurrency_benchmark.json \
  --local-root /home/ubuntu/openact-staging/mmlu_full_20260916 \
  --output /lambda/nfs/dami/openact/runs/mmlu_full_20260916
```

Do not launch a duplicate while that tmux session is alive. Read the job's
`job_status.json` for completed/published counts and `stage`; use its log for
progress inside the current shard. `waiting_for_math` is an intentional queue
state. The progress bar's `Acc` is collection success, not MMLU answer accuracy.

## Validation before launch

- 64 local staging, scheduling, matrix, and configuration tests passed.
- The initial 20 staging/scheduling tests also passed on the GPU host.
- An additional tiny real-model test covers MMLU collection, per-sample gate
  checks, exact activation replay, evaluation, verified copying, and reopening
  the persisted shard.

## Observed benchmark results

The controlled benchmark completed successfully from **15:12:44 to 15:29:00 UTC
on 2026-09-16**. Production MATH resumed automatically and its process was then
confirmed running on the GPU. The comparison covered 120 total generated
examples across the standalone and paired runs. Every run passed activation
validation and evaluation. Exact same-shape replay maximum error was zero;
paired versus standalone generated token IDs matched in every comparison.

| MATH model paired with 1B MMLU | Serial seconds | Parallel seconds | Combined throughput gain | Total peak GPU GiB |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 124.40 | 92.56 | 34.4% | 9.45 |
| Qwen2-7B | 257.92 | 220.60 | 16.9% | 21.59 |
| Llama-3-8B | 163.22 | 126.94 | 28.6% | 22.54 |

All three pairings passed admission. Peaks include the extra paused production
1B model, and even the largest observed peak left 17.46 GiB free. These are
observed pilot peaks, not guaranteed worst-case peaks over the full dataset.

The individual MATH worker took approximately 30.6%, 7.9%, and 15.7% longer in
these paired tests, respectively. Increased combined throughput does **not**
mean that the original MATH-only completion estimate remains unchanged.

The full machine-readable evidence, including exact token IDs and worker memory
peaks, is in `reports/concurrency_benchmark.json`. The final nine scheduling and
real-model MMLU integration tests also passed on the GPU host before launch.

## Production launch observed

The full MMLU job started in tmux `openact-mmlu-full` at **2026-09-16 15:31:23 UTC**,
PID **87185**. At 15:32:16 UTC it was collecting the first 100-row Llama-3.2 shard,
with a successful first sample and no errors. Its plan requires all 42,126 rows
and confirms all 57 subject groups. The existing MATH process, PID **43096**, was
simultaneously progressing through Llama-3.2 rows 3,100–3,199, without errors.

All files in the original MATH job's code checksum manifest still matched; no
running MATH package/launcher code was changed. The approximate 976-second
benchmark pause is recorded separately on the GPU at
`runs/math_full_20260916/events/concurrency_benchmark_pause_20260916.json`.

This is a launch record, not a live status or full completion report. Both jobs
continue independently of the SSH connection. Runtime state is authoritative in
each job's `job_status.json`, log, verified shard directories, and final `_SUCCESS`.
