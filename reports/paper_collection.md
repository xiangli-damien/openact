# Collection against the supplied HSS paper

Reviewed: *Hidden States as States: Discretizing Hidden-State Geometry in Language
Model Computation*, supplied NeurIPS 2026 draft, especially §3.1, §5.1, §5.2,
Table 1, Table 3, and Appendix A.2 (pages 3, 7–8, 13).

## Result

### Current experiment choices (user update)

Use the full MMLU test set (14,042), and the full TheoremQA set (800). TheoremQA
image questions are retained with `has_image` and `image_input_policy` metadata;
these text-only models receive the question text, not the image. BELEBELE uses
Qwen2-7B-Instruct for English, German, Chinese, Arabic, and Spanish, 900 examples
each (4,500 total). Safety collection is deferred. These choices supersede the
unresolved subset questions below and deliberately differ from the draft counts.
Use [capability.toml](../configs/capability.toml) for full datasets; the research
configuration retains a 20-example smoke limit.

The collector supports the paper's three representation inputs: prompt-last,
generation-average, and prefix-average, across every layer including embeddings.
The [research configuration](../configs/research.toml) now collects all layers,
both sides of final RMSNorm, every generated token, and exact prompt token IDs.
Its default task is MATH, with a 20-sample limit for an initial run.

CPU numerical tests check the Llama 2, Llama 3, Llama 3.2 (including its scaled
RoPE), and Qwen2 implementations, with eager attention and SDPA. These use tiny
random models. Actual model tokenizers/configurations were checked separately.
**Full-size GPU execution and exact reproduction of the paper's results remain
unverified.** The GPU acceptance command below verifies the actual checkpoints.

Verification: 197 tests passed / 12 existing-run tests skipped on Transformers
5.17.0; all 53 updated runtime/config tests also passed on 4.57.6. The dataset
and tokenizer audit passed 1,488 prompt/chat checks across the four real tokenizers
and scanned 45,196 rows across 41 configurations. See [test_results.json](test_results.json)
and [data_and_prompts.json](data_and_prompts.json).

## Models and expected saved shapes

Here `T` includes all generated token IDs, including any generated EOS. These
dimensions were read from the official checkpoint configurations; immutable
revisions are recorded in [data_and_prompts.json](data_and_prompts.json).

| Model ID | Decoder blocks | Saved layers (embedding included) | Per-response tensor | Paper use |
|---|---:|---:|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 16 | 17 | `(T, 17, 2048)` | Default map; cross-model |
| `Qwen/Qwen2-7B-Instruct` | 28 | 29 | `(T, 29, 3584)` | Reliability; cross-model; reasoning prediction; monitoring; characterization |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 33 | `(T, 33, 4096)` | Cross-model; reasoning prediction |
| `meta-llama/Llama-2-7b-chat-hf` | 32 | 33 | `(T, 33, 4096)` | Safety prediction |

Llama Guard is a response-labeling judge, not one of these activation-collection
models. The user-selected original `meta-llama/LlamaGuard-7b` remains provisional;
its gated tokenizer was inaccessible during the earlier audit.

## Position and layer conventions

For prompt length `P` and generated length `T`, the full token sequence has
positions `0..P+T-1`:

- Prompt-last is position `P-1` of the **complete chat-formatted model input**.
  It can be an assistant-header/control token rather than the final word of the
  user's question. It is saved separately from generated activations.
- Generated row `i` is the state at position `P+i`, after consuming generated
  token `i`. The final generated token gets a state, including EOS when generated.
- Generation-average is the mean of rows `0..T-1` only. Prompt-last is excluded.
- Prefix-average at an exclusive token boundary `b` is the mean of rows `0..b-1`.
  No state from a later generated token is included.
- Hidden index `0` is the embedding output; indices `1..L-1` are the intermediate
  block outputs. HF's last hidden index `L` is **after final RMSNorm**.
  `final_norm/pre` additionally stores the final block output before that norm.

Generation chooses the exact token IDs first. A separate causal teacher-forced
forward over those IDs collects states (`eval()`, inference mode, no KV cache).
This follows the user's requested extraction method. Tests compare the saved
prompt-last vector with a prompt-only forward, and **every** saved response row
with a forward ending at that token. This checks causal equivalence and catches
off-by-one errors independently of the full-sequence extraction implementation.

The paper does not say whether the final layer is before or after RMSNorm. Use
either convention consistently across samples; save both now to preserve the
choice. No PCA, standardization, or additional vector normalization is applied.
The float32 storage preserves bf16 inference values; it does not increase their
original numerical precision.

## Read all three representations

```python
from openact_core import Run

sample = Run("runs/math_qwen2")[0]
prompt_ids = sample.prompt_token_ids             # exact chat-formatted input
response_ids = sample.token_ids                 # includes generated EOS
prompt_last = sample.prompt_last_hidden_states  # (L+1, H)
tokens = sample.hidden_states                   # (T, L+1, H)
generation_avg = sample.mean_hidden_states      # (L+1, H)

before_rms = sample.get_final_norm_states("pre")   # (T, H)
after_rms = sample.get_final_norm_states("post")   # (T, H)
prompt_before_rms = sample.get_final_norm_states("pre", "prompt_last")

# b is the exclusive token boundary supplied by your sentence segmentation.
b = min(10, sample.n_tokens)
prefix_avg = tokens[:b].mean(axis=0)             # (L+1, H)
prefix_before_rms = before_rms[:b].mean(axis=0)   # (H,)
```

For a consistently pre-final-norm all-layer representation, copy `tokens` and
replace only `tokens[:, -1, :]` with `before_rms`; do the corresponding replacement
for prompt-last and means. Do not treat pre/post norm as two consecutive decoder
layers. Sentence segmentation, train/test fitting, GMM/ICL, state alignment, and
FAR calibration remain downstream analysis. Current generation metrics are
whole-response aggregates, not the paper's sentence-prefix entropy/logprob baselines.

## GPU acceptance and collection commands

From the repository root, with access to the gated Meta checkpoints and a CUDA
PyTorch installation on the A100/H100:

```bash
uv sync --frozen --extra dev
uv run openact config-check configs/research.toml
uv run python scripts/check_paper_collection.py --output runs/paper_check
```

The acceptance script loads the four models **sequentially** onto `cuda:0` in
bf16, collects two short zero-shot CoT examples per model, reopens their saved
runs, and compares all layers, both RMSNorm sides, prompt-last, and every generated
token with independent causal-prefix forwards. It also checks token IDs, means,
chat formatting, dimensions, and storage validity. It writes `verification.json`
with checkpoint revisions, measured errors, and peak CUDA allocation. A failure
exits nonzero. Numerical tolerance is a maximum 3% relative L2 vector error to
allow bf16 kernel variation across sequence lengths; it is recorded in the report.

Select one model with `--model qwen2` (also `llama32`, `llama3`, `llama2`). The
default 16-token check is a correctness smoke test, not a long-context memory test.
Use fresh output paths for each invocation.

Example MATH collection after the check:

```bash
uv run openact collect --config configs/research.toml \
  --model Qwen/Qwen2-7B-Instruct --task math --max-samples 5000 \
  --output runs/math_qwen2
uv run openact-validate runs/math_qwen2 --strict
```

Change `--model` to the Llama 3.2 or Llama 3 ID for their MATH runs. Safety:

```bash
uv run openact collect --config configs/research.toml \
  --model meta-llama/Llama-2-7b-chat-hf --task jbb --template zot \
  --max-samples 200 --output runs/jbb_llama2
```

This JBB command uses all 200 registered goal prompts, not a claimed reconstruction
of the paper's unspecified goal/artifact selection. The `zot` safety wrapper now
actually appends `Please reason step by step.`; `raw` preserves the original
prompt. The paper does not provide the exact CoT wording.

All-layer storage is substantial. Uncompressed float32 per-token activations,
including separate pre/post RMS arrays, cost about 156 KB/token (Llama 3.2),
444 KB/token (Qwen2), or 573 KB/token (Llama 3/2). At 5,000 × 2,048 generated
tokens this is roughly 1.59/4.55/5.87 TB respectively, before compression and
small metadata/mean overheads. Actual size depends on response lengths and
compression. The current 2,048-token cap is a project default, not a value stated
in the paper; inspect `finish_reason="length"` when choosing your run budget.

## What prevents an exact reproduction claim

| Paper detail | Current status |
|---|---|
| MATH, 5,000 examples | Canonical evaluation categories total 5,000; supported. |
| MMLU, 14,000 examples | User selected all 14,042 test rows for this experiment. |
| BELEBELE, 2,100 across English/German/Chinese | User selected Qwen2 with five full languages (en/de/zh/ar/es), 4,500 rows total. |
| TheoremQA | User selected all 800 rows, including 53 image questions; image presence is marked and only the text is passed to these models. |
| JailbreakBench | Supported, but exact goal/artifact mix is unspecified. Response safety labels require the chosen Guard model; prompt harmful/benign labels are not response labels. |
| HarmBench | No built-in adapter. Supply the paper's exact rendered rows using `--task prepared --prepared-path ...`; there is no automatic substitute benchmark. |
| Prompts, checkpoints, EOS means, token cap | Exact prompt strings, model revisions, terminal-token policy, and max generation length are not specified in the paper. Current explicit choices are saved in run metadata. |
| Prefix monitoring | Raw prefix states are recoverable. Boundary segmentation and sentence-level metric baselines are not implemented. |

Author-provided prompts, dataset IDs/subsets, and final-norm/EOS conventions are
needed to close these fidelity gaps. The collecting procedure itself is tested
without inferring those missing experiment choices.
