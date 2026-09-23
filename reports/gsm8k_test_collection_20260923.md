# GSM8K test: Qwen2 and Llama-3

User-approved scope: official `openai/gsm8k`, `main/test`, **1,319 questions per model**. Models are `Qwen/Qwen2-7B-Instruct` and `meta-llama/Meta-Llama-3-8B-Instruct`. This excludes GSM8K train and Llama-3.2-1B. Model/dataset commits are pinned in `configs/gsm8k_test.toml`.

## Isolation and outputs

- Git branch: `codex/gsm8k-test-qwen2-llama3`, separate worktree; main and HSS sources unchanged.
- Qwen complete existing data: `/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923/qwen2`. Reuse only after all-file SHA checks, full 1,319-row coverage checks, and independently rescored labels pass.
- New queue/results: `/lambda/nfs/dami/openact/runs/gsm8k_test_20260923`.
- New full Llama shards: `llama3_full/llama3/shard_*` inside that directory. Two-question smoke is separate.
- SSD staging: `/home/ubuntu/openact-gsm8k-llama3-20260923`. Each 32-question shard is evaluated and causally replay-checked, copied to dami and SHA-verified, then its staging copy is removed. Retain at least 50 GiB SSD free.

## Exact collection contract

Same MATH-matched zero-shot CoT prompt, boxed numeric answer, each model's own pinned chat template. HF bf16, SDPA, greedy, seed 42, 2,048 maximum new tokens. Preserve model-specific inherited repetition penalties (Qwen 1.05; Llama 1.0), asserting and recording defaults at load. Thus this matches earlier collection; it is not a new raw-greedy Qwen protocol.

Generate first; use teacher-forced forward on the actual prompt and answer to extract all layers. Save prompt-last, every generated token, and generation mean, plus final RMSNorm pre and post. Raw values, float32 storage preserving bf16 values; no analysis normalization. The prompt-last position includes chat-template delimiters. Generated-token means include EOS/special tokens. A 2,048-token capped answer remains in the dataset and is flagged, not silently excluded.

Stored fields include original question, answer, gold numeric answer, extracted answer/correctness label, token IDs, metadata, tensors and checksums. Qwen has 29 hidden-state entries/3,584 dimensions; Llama has 33/4,096, with final pre-norm separately saved.

## Run and inspect

From the isolated remote worktree, using the existing OpenAct environment:

```bash
/lambda/nfs/dami/openact/.venv/bin/python scripts/run_gsm8k_test_queue.py
```

The queue first audits/reuses Qwen, waits for GPU idleness, runs the two-row real Llama smoke and audit, then full Llama and audit. An output lock prevents duplicate queues. A failed stage stops the queue and preserves artifacts. Resumption uses unchanged plan hashes and shard receipts; do not edit running source/configuration.

Read `queue_status.json`, `llama3_full/job_status.json`, and stage logs. `collection_manifest.json` appears only after both complete audits pass. Existing Qwen data is referenced by path, not duplicated or relabelled. The collection audit rehashes every stored file and re-scores every answer, verifies original GPU causal-replay receipts; it does not claim to perform a second full GPU replay.

HSS GPU functional work must wait while this queue owns the GPU; CPU code preparation and analysis may continue. No new GPU rental or environment upgrade is needed.

## Execution amendment

18:39 UTC: Qwen audit passed all 1,319 rows/42 shards, all-file SHA and independent answer re-scoring. 341,885 generated tokens; 1,134 correct (86.0%); 57,055,267,394 bytes including transfer receipts.

The initial Llama smoke stopped **before generating any samples**: Transformers 5 leaves an omitted repetition penalty as `None` until `generate()` fills global defaults. The configuration check now uses that same fallback rule and saves both configured and resolved defaults; the generation protocol is unchanged. A regression test covers omitted versus explicit repetition penalty. Preserve the old failure and launch the corrected queue under `/lambda/nfs/dami/openact/runs/gsm8k_test_20260923_v2`, staging `/home/ubuntu/openact-gsm8k-llama3-20260923-v2`. Its `--reuse-qwen-audit-sha256` points to the exact already-completed Qwen audit, avoiding a redundant 53 GiB scan. Old and new collection plans remain separate.

18:44 UTC: corrected queue PID444653, source `bff452b`. Five tests passed locally and remotely. Llama smoke **2/2 complete and independently audited**, including 33×4096 all-layer arrays, both final norm sides, numeric labels and full-file SHA. All1,319 prompts fit the8,192-token context (largest212 plus2,048 output budget). Full collection launched PID444718 at18:44UTC; observed15,870MiB GPU memory and real collection progress. Full-run fingerprint `b93ea7101336f581db1364c7fa7124cfc4df0901d5c93c0665efcaa00f52ca43`. This is launch evidence, not a full-completion claim; check the current status files.
