# Three-model full MATH collection

## Scope and settings

Authorized on 2026-09-16: MATH test, all 5,000 unique questions for each of:

- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen2-7B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`

Total: **15,000 generations**. No MMLU, BELEBELE, TheoremQA, safety, Llama-2,
or Llama Guard jobs are included. Exact model revisions are pinned in
`configs/math_full.toml`; source checksums and effective settings are recorded in
the GPU job's `job_plan.json`.

HF Transformers, bf16, SDPA, greedy zero-shot CoT, up to 2,048 generated tokens.
The separate teacher-forced pass captures embeddings and all HF hidden layers,
plus both sides of the model's final RMSNorm. Prompt-last, every emitted token
(including an emitted EOS), and generation means are retained as float32 raw
values. Length-capped generations are explicitly marked. Each shard also gets
MATH correctness labels.

## Storage and restart behavior

GPU: A100-SXM4 40GB on `gpu2`. At launch preparation the root disk had about
406 GiB free. Earlier three-sample pilots estimated about 800 GB for this scope,
so the whole collection cannot safely remain on the instance's root disk.

`scripts/run_math_collection.py` keeps each model loaded while collecting
contiguous 100-question shards to local disk. A single background worker copies
completed shards into dami; at most two copies are outstanding. Before starting
a shard, the launcher checks a conservative uncompressed size allowance plus
a 64 GiB free-space reserve, waiting for transfers if needed.

Each shard is fully validated and evaluated before transfer. Copying flushes
files with `fsync`, compares all file sizes and SHA-256 digests, then publishes
the directory by rename. Only after that does it remove the local duplicate.
An interrupted copy never becomes a completed shard. Failed collection attempts
are retained under `.failed-*`; incomplete transfers remain `.incoming-*`.

Repeating the same command checks data/code/config fingerprints and verifies
already-published shards before skipping them. A partially collected shard is
restarted; this is shard-boundary recovery, not token-level resume. Collection
attempts and transient transfer errors are retried up to three times. Persistent
errors stop the job and are written to `job_status.json`.

## Paths on gpu2

- Prepared MATH: `/lambda/nfs/dami/openact-data/prepared/math`
- Full local staging: `/home/ubuntu/openact-staging/math_full_20260916`
- Full persistent output: `/lambda/nfs/dami/openact/runs/math_full_20260916`
- Full log: `/lambda/nfs/dami/openact/logs/math_full_20260916.log`
- Smoke output: `/lambda/nfs/dami/openact/runs/math_staging_smoke_20260916`
- Smoke log: `/lambda/nfs/dami/openact/logs/math_staging_smoke_20260916.log`

Each model's final data consists of 50 standalone OpenAct Runs, for example:

```text
math_full_20260916/
  job_plan.json
  job_status.json
  llama32/shard_00000_00100/
  llama32/shard_00100_00200/
  ...
  qwen2/shard_00000_00100/
  ...
  llama3/shard_04900_05000/
  _SUCCESS                        # created only when all 15,000 are published
```

Read a completed shard directly with `Run(path)`. `sample_idx` restarts within
each shard; use stable `sample_id` to join across shards/models. The shard's
`_SHARD.json` records its ordered original row range and IDs, and
`_COPY_VERIFIED.json` records transfer checksums. Do not modify the published
shards in place; write subsequent analysis into a separate output directory.

## Run and inspect

From `/lambda/nfs/dami/openact` on gpu2:

```bash
.venv/bin/python -u scripts/run_math_collection.py \
  --local-root /home/ubuntu/openact-staging/math_full_20260916 \
  --output /lambda/nfs/dami/openact/runs/math_full_20260916
```

This command is run in detached tmux session `openact-math-full`, with output
redirected to the full log above, so closing SSH or the Mac does not stop it.
Do not start a duplicate command while that session is running.

```bash
tail -f /lambda/nfs/dami/openact/logs/math_full_20260916.log
cat /lambda/nfs/dami/openact/runs/math_full_20260916/job_status.json
```

`job_status.json` counts only published shards. The log reports current
within-shard collection progress. The progress bar's `Acc` means successful
collection fraction; it is not MATH answer accuracy.

## Validation

- Local: 56 staging, matrix, and configuration/integrity tests passed.
- GPU: the same 56 tests passed before collection.
- The staging smoke **passed 9/9 samples**, including one 2,048-token capped
  Llama-3 response. Every sample's collection and MATH evaluation succeeded;
  all three shards were copied, SHA-256 checked, and published on dami.
  `reports/math_staging_smoke.json` is the post-resume snapshot. The original
  local write and background transfer observations were:

  | Model | Tokens (3 rows) | Collection seconds | Transfer seconds | Compressed bytes |
  |---|---:|---:|---:|---:|
  | Llama-3.2-1B | 699 | 11.62 | 4.02 | 40,510,771 |
  | Qwen2-7B | 975 | 27.81 | 9.53 | 162,026,832 |
  | Llama-3-8B | 2,235 | 69.12 | 20.23 | 472,943,636 |

  Transfers overlap other work. These first three questions are not a
  representative benchmark or a reliable full-dataset timing forecast.
- Running the identical smoke command again completed with 9/9 published rows
  and **zero new generations**. It verified existing checksums/IDs and skipped
  all completed shards. The resume log contains no collection progress bars.
- For every production shard: exact ordered IDs, successful complete rows,
  activation shapes/finite values/means, and final-norm output equality with the
  last HF hidden entry. The first and longest response are deeply replayed with
  exact same-shape activation and causality checks before copying.

Full completion must be established from the full job's `_SUCCESS` and all
15,000 published sample IDs, not from the smoke test or the process launch.

## Launch

The full job was launched in `openact-math-full` at **2026-09-16 06:46:13 UTC**.
Runtime state belongs to the GPU `job_status.json` and log; this document is a
launch record, not a live completion report.
