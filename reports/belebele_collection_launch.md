# BELEBELE full collection: Llama-3.2-1B and Qwen2-7B

Authorized 2026-09-25: two models, English `eng_Latn`, German `deu_Latn`, simplified Chinese `zho_Hans`; 900 test rows per language and model, 5,400 full responses. Fixed dataset commit `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`. No fitting or steering in this job.

`configs/belebele_full.toml` pins both model revisions. Raw passages/questions/options remain in their source language. Existing `zot` instructions are English, request step-by-step reasoning and final `Answer: A/B/C/D`. Both models use their native chat template. Generation matches the existing MATH/MMLU OpenAct configuration: BF16 HF transformers, greedy, seed42, maximum2048 new tokens; inherited pinned model generation defaults are saved explicitly. Qwen's historical repetition penalty1.05 is preserved, not silently changed to the steering setting1.0.

Capture uses a separate teacher-forced forward over exact prompt and generated IDs: every generated token at every layer (including embedding and the last generated token), prompt last at every layer, mean vectors, final RMSNorm pre/post per-token/prompt-last/mean. Stored float32 retains original BF16 values without additional normalization. Prompt tokens other than the final prompt token are not stored, matching the existing full collection contract. All original text, token IDs, offsets and correctness labels are retained.

Launch after local tests, GitHub push and Lambda fetch/fast-forward:

```bash
cd /lambda/nfs/dami/openact
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_belebele_collection.py
```

The runner first audits all2,700 prepared rows and freezes source/data hashes. It then collects and verifies first/last examples for all six model/language cells (12 smoke responses). Only after every smoke cell passes does the5,400-response full queue run, Llama then Qwen, each English/German/Chinese. Each model loads once per phase. No concurrent GPU model jobs.

SSD staging `/home/ubuntu/openact-belebele-20260925`; persistent root `/lambda/nfs/dami/openact/runs/belebele_full_20260925`. Full OpenAct runs are at `full/{llama32,qwen2}/belebele_{en,de,zh}/shard_*`; smoke is separate. Shards contain at most32 rows; worst-case space check preserves50GiB free. Each shard validates complete IDs/arrays, RMS sides and means, prompt-last shapes, two exact teacher-forced and causal replay samples, and all automatic evaluation labels. Publication verifies every file's SHA256 before deleting only that identical task-owned staging shard. Partial failed attempts are retained; errors stop and are recorded. Repeating the unchanged command resumes at verified shard boundaries. Root `job_status.json` reports phase/current shard/durable progress, `job_plan.json` freezes provenance, root `_SUCCESS` alone confirms all5,400 full rows finished.

Status: launch must be confirmed by the remote PID/status and real smoke results; this protocol file alone is not evidence of a running or completed job.
