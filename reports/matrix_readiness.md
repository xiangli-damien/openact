# Full matrix readiness and cost estimate

Verified on 2026-09-16 UTC, Lambda `gpu2`, **A100-SXM4 40GB**, bf16 inference.

## Result

**25/25 model × dataset/language cells passed collection, activation verification,
and persisted evaluation.** There are 3 deterministic examples per cell, **75 real
GPU generations**, with a 2,048-token maximum and greedy zero-shot CoT (`zot`).
No full-dataset activation job has been started.

- All 75 samples have complete token/layer arrays, finite values, consistent
  means, and saved final-norm outputs equal to the last HF hidden layer.
- The first and longest response of each cell were deeply checked: **45 samples**.
  Every stored generated token and the prompt-last state were compared with a
  new full-sequence forward, including both final RMSNorm sides. **Maximum error: 0.**
- Future token IDs were replaced while keeping tensor dimensions fixed. Prompt-last
  and selected response positions (first and eighth tokens where present) stayed
  exactly unchanged. This independently checks causal alignment without mixing
  it with different-shape bf16 rounding.
- Variable-length prefix forwards are also recorded. Their maximum relative L2
  difference is **7.39%**. Four cells exceeded the initial arbitrary 3% threshold;
  all passed exact replay and same-shape causal perturbations. The original errors
  are preserved in the report. We did not increase the old threshold to hide them.
- The longest tested response was **2,048 tokens**, in Llama-3.2/TheoremQA. It reached
  the token cap (1/75 samples); this is a completed capture of a capped response,
  not proof that the model finished its natural answer.
- Full-dataset chat-template/token-length checks covered all **140,218 planned
  model inputs**. No prompt needed truncation or a reduced generation budget.
- Highest recorded pilot CUDA allocation: **16.56 GB**. This is observed pilot
  usage, not a worst-case guarantee for every full-set response.
- **213 local tests passed**, 12 optional existing-run tests skipped. **71 runtime,
  configuration, and matrix tests passed on Lambda**. The final safety grouping
  regression also passed. A separate two-sample Qwen2 GPU check passed after the
  NFS writer changes, exercising the current writer and short-check entry point.
- All **22,542 capability reference answers** passed the parser/matcher format
  audit. This checks label plumbing and reference formats, not complete semantic
  equivalence to an official benchmark evaluation harness.

Evidence: [matrix smoke](matrix_smoke.json), [current writer GPU check](current_writer_check.json),
[tests](test_results.json), [reference answer audit](evaluation_reference_audit.json),
[prepared data audit](matrix_datasets.json), [cost calculations](matrix_cost.json).

## Confirmed experiment

| Dataset | Rows per selected model | Models |
|---|---:|---|
| MATH test, all seven categories | 5,000 | All four |
| MMLU test, all 57 subjects | 14,042 | All four |
| BELEBELE English / German / Chinese | 900 each (2,700 total) | All four |
| TheoremQA | 800 | All four |
| WildJailbreak train, exact `vanilla_harmful` filter | 50,050 | Llama-2 only |

Total: **140,218 generations**. Prepared input tables contain no empty prompts or
duplicate sample IDs. Every selected row uses `zot`. Three examples per cell were
selected deterministically, shared across models for each dataset.

TheoremQA retains all **53 image questions**, marked by metadata. The selected
models receive question text only; this does not reproduce an image-enabled test.
WildJailbreak uses only the source `vanilla` prompt, never reference `completion`
or adversarial rewrites. Source: [official WildJailbreak dataset](https://huggingface.co/datasets/allenai/wildjailbreak).

Generation models and immutable revisions are in
[experiment_matrix.toml](../configs/experiment_matrix.toml). All selected dataset
revisions, splits, configs, and prompt templates are recorded in the prepared
manifests and copied into [matrix_datasets.json](matrix_datasets.json).

### Safety judge

Confirmed: **`meta-llama/Llama-Guard-3-8B`**, revision
`7327bd9f6efbbe6101dc6cc4736302b3cbb6e425`. It loads after generation models are
unloaded, uses its own chat template, and labels assistant responses as safe or
unsafe. All three pilot labels were parsed and saved without errors. Their labels
are `safe`; this tiny smoke set is not an estimate of the model's safety rate.

Safety labels come from Guard, while refusal is a separate heuristic field.
Missing benign splits have `null` rates. Category rates group by the dataset's
input category. Label probabilities use the actual classification token's logits,
allowing leading newlines; unsupported tokenizations have no probability.
These probabilities are not calibrated confidence estimates.

## What is saved

1. Generate response token IDs with greedy HF Transformers decoding.
2. Run a separate teacher-forced forward over the complete prompt + response,
   without a KV cache. Generation's hidden-state outputs are not used.
3. Save the last chat-formatted prompt token, every generated token (including EOS
   when emitted), generation mean, all HF hidden layers including embeddings,
   and final RMSNorm **input and output**, each with prompt-last/token/mean forms.
4. Preserve raw bf16 values in float32 arrays with lossless Zarr compression.
   No vector normalization, PCA, or standardization is applied.

HF's last hidden entry is after final RMSNorm; `final_norm/pre` separately contains
its input. Response row `i` is the state after consuming response token `i`.
Prompt-last may be an assistant header/control token, depending on the model's
chat template. Prefix means can be computed from the saved token states.

The paper does not specify every RMS/EOS/prompt/token-cap convention. The expanded
matrix, Guard 3 selection, text-only TheoremQA handling, and 2,048 cap are explicit
experiment choices. Successful collection is not a claim that paper metrics have
been reproduced. GMM/ICL fitting, alignment, sentence segmentation, and FAR
calibration remain external CPU analysis; their settings are recorded in TOML.

## Time and disk estimate

Measured on this A100 40GB, **one example at a time**, writing to the persistent
Lambda filesystem. Each cell's observed seconds/sample and compressed bytes/sample
were multiplied by its full row count. Judge loading is separated from per-sample
inference; stored-array scans are included, and first/longest-sample deep checks
are counted once per cell.

| Model | Total hours | Days | Compressed TB |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 47.1 | 1.96 | 0.38 |
| Qwen2-7B-Instruct | 88.8 | 3.70 | 1.04 |
| Llama-3-8B-Instruct | 79.0 | 3.29 | 0.80 |
| Llama-2-7B-Chat (including Safety) | 367.8 | 15.32 | 4.85 |
| **Total** | **582.6** | **24.28** | **7.07** |

Breakdown: **564.6 h collection**, **6.3 h evaluation**,
**11.7 h validation**. The Safety cell alone is approximately
266.9 h / 11.1 days and 3.59 TB. The capability matrix alone is approximately
315.7 h / 13.2 days and 3.48 TB.

With **50% planning headroom**: **874 h / 36.4 days**,
**10.6 TB**. This headroom is not a statistical confidence interval.

Limitations of these estimates:

- Only three examples per cell; answer lengths and refusal behavior vary greatly.
- One pilot answer hit the cap. A larger token budget can substantially increase
  time and size. Download/model-loading time, interruptions, retries, and fitting
  are excluded. No inference batching or vLLM acceleration was measured.
- NFS cache state and provider throughput can change. The current writer avoids
  large multi-sample summary chunk rewrites; dominant token-array layout is unchanged.
- Sizes are compressed file bytes, decimal TB. Quotas, billing, filesystem overhead,
  and backup copies are separate. The mount's virtual 8 EiB display is not a quota.

Illustrative length scenarios (all responses set to the same length; **collection
only**, excluding evaluation/validation; fixed measured seconds and bytes/token):

| Tokens per response | Collection days | Compressed TB |
|---:|---:|---:|
| 128 | 11.3 | 3.25 |
| 256 | 22.7 | 6.51 |
| 512 | 45.4 | 13.02 |
| 1024 | 90.7 | 26.03 |
| 2048 | 181.5 | 52.07 |

These scenarios expose sensitivity to response length; they are not forecasts of
natural model completion lengths.

### Per-cell planning figures

| Model | Dataset | Full rows | Pilot generated tokens | Total hours | Compressed GB |
|---|---|---:|---|---:|---:|
| llama32 | math | 5,000 | 235, 883, 235 | 15.1 | 129.2 |
| llama32 | mmlu | 14,042 | 266, 44, 334 | 22.9 | 176.1 |
| llama32 | belebele_en | 900 | 234, 130, 213 | 1.5 | 10.1 |
| llama32 | belebele_de | 900 | 285, 420, 394 | 2.2 | 19.1 |
| llama32 | belebele_zh | 900 | 36, 135, 310 | 1.2 | 8.5 |
| llama32 | theoremqa | 800 | 279, 2048, 263 | 4.3 | 39.5 |
| qwen2 | math | 5,000 | 639, 848, 297 | 36.7 | 491.0 |
| qwen2 | mmlu | 14,042 | 191, 72, 240 | 38.2 | 396.5 |
| qwen2 | belebele_en | 900 | 234, 177, 106 | 2.7 | 26.2 |
| qwen2 | belebele_de | 900 | 166, 55, 323 | 2.8 | 27.6 |
| qwen2 | belebele_zh | 900 | 187, 54, 374 | 3.2 | 31.1 |
| qwen2 | theoremqa | 800 | 252, 745, 492 | 5.1 | 65.9 |
| llama3 | math | 5,000 | 122, 275, 105 | 18.3 | 179.3 |
| llama3 | mmlu | 14,042 | 90, 127, 182 | 43.2 | 404.3 |
| llama3 | belebele_en | 900 | 303, 164, 211 | 3.5 | 43.7 |
| llama3 | belebele_de | 900 | 293, 203, 307 | 4.0 | 51.6 |
| llama3 | belebele_zh | 900 | 180, 223, 404 | 4.7 | 51.8 |
| llama3 | theoremqa | 800 | 230, 667, 356 | 5.2 | 70.6 |
| llama2 | math | 5,000 | 455, 262, 245 | 25.1 | 340.4 |
| llama2 | mmlu | 14,042 | 177, 183, 361 | 59.4 | 720.9 |
| llama2 | belebele_en | 900 | 184, 168, 198 | 3.3 | 35.4 |
| llama2 | belebele_de | 900 | 370, 278, 460 | 5.6 | 70.7 |
| llama2 | belebele_zh | 900 | 182, 145, 455 | 4.1 | 50.1 |
| llama2 | theoremqa | 800 | 44, 408, 298 | 3.4 | 42.4 |
| llama2 | wildjailbreak | 50,050 | 520, 218, 271 | 266.9 | 3587.2 |

## Persistent locations and reproduction

- Active repository: **`/lambda/nfs/dami/openact`** (`~/dami/openact`).
- Prepared data: **`/lambda/nfs/dami/openact-data/prepared`**.
- This pilot: **`/lambda/nfs/dami/openact/runs/matrix_smoke_20260916`**.
- Authoritative pilot report: `matrix_report_rechecked.json` in that directory.
  The original `matrix_report.json` is retained with the initial diagnostic errors.
- Latest writer check: `runs/current_writer_check_20260916`.
- Reports and logs are also under the persistent checkout. The environment and
  downloaded weights are on the instance disk and can be recreated from `uv.lock`.

Code and reports are pushed to the existing GitHub repository and fetched into
this persistent checkout. See [Lambda setup](lambda_gpu_setup.md) for SSH, environment,
smoke, recheck, and full-run commands. Full outputs require fresh directories;
resume is not implemented. Verify the actual `dami` storage quota before a full run.

## Fixes made during this matrix audit

- Added pinned WildJailbreak Vanilla Harmful loading and the complete four-model matrix.
- Made TheoremQA evaluation use each row's declared answer type.
- Preserved `\text{...}` answer contents; this fixed 68 MATH reference false negatives.
- Prevented silent prompt truncation and recorded effective generation context budgets.
- Replaced arbitrary bf16 prefix-threshold failures with exact same-shape replay
  and future-token perturbation checks, retaining numerical diagnostics.
- Corrected Guard score positioning, safety/refusal semantics, missing-split rates,
  and input-category aggregation.
- Avoided repeated activation reads during verification, large summary-chunk
  rewrites, and per-pointer NFS reads during finalization.
