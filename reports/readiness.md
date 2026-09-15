# OpenAct readiness audit — 2026-09-15

## Result

The collection path passes CPU numerical, storage, CLI, dataset, tokenizer,
and package-build checks. **Qwen2-7B-Instruct has now passed a short full-size
CUDA/bf16 collection check on A100 40GB.** Other checkpoints and long-context
capacity remain unverified; see [the Lambda setup report](lambda_gpu_setup.md).

The follow-up [paper collection audit](paper_collection.md) checks the supplied
paper's layer/token conventions and model matrix. The research TOML now defaults
to MATH and saves **all layers**, with separate prompt-last and generated states.

The supplied [research TOML](../configs/research.toml) uses your requested
teacher-forced extraction, greedy decoding, zero-shot CoT, bf16 model inference,
and raw activation storage. It records diagonal GMM, ICL selection with `k_max=80`,
cosine Hungarian alignment with `eta=0.6`, stratified 40/60 fit/evaluation split,
Laplace smoothing (`alpha=1`), and a 10% FAR target. The 40% partition is interpreted
as the fitting partition. Seed 42 and alpha 1 are explicit defaults.

GMM/ICL fitting, state alignment, split generation, and FAR calibration are **not
implemented in this repository**. Those TOML tables are validated metadata for
your downstream CPU analysis. The collection CLI does not claim to execute them.
vLLM is also not integrated; the configured backend is Transformers.

## Activation semantics

1. Generate the response without collecting generation hidden states.
2. Release the generation KV cache.
3. Run prompt plus generated token IDs through the model with teacher forcing,
   `use_cache=False`, and `torch.inference_mode()`.
4. Save the state at each response token's own position, plus prompt-last.
5. Capture the final norm module's input **and** output. Do not apply any additional
   normalization to the stored vectors.

The previous implementation attached the final prompt token's state to the first
response token and omitted the last response token's own state. A tiny real Llama
reproduction confirmed that shift. Existing runs are not repaired automatically;
recollect data if it was produced with that implementation.

For Llama and Qwen2, `hidden_states[..., -1, :]` is the output of final RMSNorm.
Its input is the final decoder block's residual stream. New storage paths are:

| Array | Shape |
|---|---|
| `final_norm/pre/per_token` | `(response_tokens, hidden_dim)` |
| `final_norm/post/per_token` | `(response_tokens, hidden_dim)` |
| `final_norm/{pre,post}/mean` | `(samples, hidden_dim)` |
| `final_norm/{pre,post}/prompt_last` | `(samples, hidden_dim)` |

The research TOML saves float32 arrays, which preserve bf16 model values exactly.
The default library storage dtype remains float16 unless overridden. HF layer
indices remain unchanged: index 0 is the embedding output; the last index is the
final norm output. Select layers on the CLI with `--layers=-1,-2`.

```python
from openact_core import Run

sample = Run("runs/gpu_smoke")[0]
before_rms = sample.get_final_norm_states("pre")
after_rms = sample.get_final_norm_states("post")
prompt_before_rms = sample.get_final_norm_states("pre", "prompt_last")
```

EOS and generated special tokens remain in token/state arrays, with zero-width
text offsets. Means therefore include these tokens. Filter those positions in
downstream analysis if a content-token-only mean is required.

## Verification performed

**197 tests passed; 12 existing-run tests skipped**, under Transformers 5.17.0.
The earlier 183-test suite also passed under 4.57.6; the updated 53 runtime/config
tests passed again under 4.57.6. See [test_results.json](test_results.json).

- Installed the uv workspace from its lockfile and built all four wheels and
  source distributions. Checked wheel imports and packaged prompt/data resources.
- Tested on Python 3.11.16, torch 2.14.0, Transformers 5.17.0, datasets 5.0.1,
  Zarr 2.18.7, numcodecs 0.13.1, NumPy 2.4.6, pandas 3.0.5, and pyarrow 25.0.1.
- Checked Transformers 4.57.6 in a separate environment (earlier full suite, then
  the updated runtime/config tests).
- Numerical checks use tiny real Llama, Qwen2, and GPT-2 models, with independent
  full-sequence reference forwards. Tests cover one/multiple response tokens,
  EOS, bf16, pre/post norm, attention/MLP capture, prompt-last, means, and storage.
- Verified config overrides, bad configuration rejection, failed-run markers,
  overwrite protection, prepared data, label persistence, and structural corruption
  detection. `openact-core` imports without loading torch.
- Ran the actual collect → validate → evaluate CLI sequence on two real GSM8K
  examples using a tiny local model. This verifies execution, not answer quality.
- Scanned **45,196 rows across 41 dataset/language configurations**, with no empty
  prompts or missing expected labels. Rendered every registered prompt variant
  on representative rows.
- Performed **1,488 prompt/chat-template checks**: 372 for each of your four
  generation models. Tested their actual HF tokenizers, generation templates,
  EOS configuration, and multilingual/emoji offsets. No weights were downloaded.

The full dataset/model details and immutable dataset revisions are in
[data_and_prompts.json](data_and_prompts.json). All 14 dataset sources are pinned
in [dataset_versions.json](../packages/openact-collect/src/openact_collect/data/dataset_versions.json).
The model's resolved revision, exact chat template, dataset source/config/split/
revision/fingerprint, extraction method, and run configuration are saved in the
run manifest. CLI overrides are reflected in the effective model/generation/
capture fields; `custom.run_config` preserves the supplied TOML.

| Dataset | Verified evaluation rows |
|---|---:|
| GSM8K, main/test | 1,319 |
| MMLU, all 57 subjects/test | 14,042 |
| MATH, all 7 categories/test | 5,000 |
| TheoremQA, all/test (53 image questions marked; text input only) | 800 |
| ARC-Challenge/test | 1,172 |
| CommonsenseQA/validation | 1,221 |
| TruthfulQA, generation/validation | 817 |
| HumanEval/test | 164 |
| IFEval/train (official evaluation prompts) | 541 |
| MGSM/test | 250 × 11 languages |
| Belebele/test | 900 × 18 configured languages |
| JBB, goal prompts | 100 harmful + 100 benign |
| AdvBench | 520 harmful |
| XSTest | 250 benign + 200 harmful |

This establishes schema, labels, source versions, and rendering compatibility.
It is not an independent re-annotation of benchmark answers or a model-quality
evaluation. HumanEval and IFEval currently use LLM judges rather than the official
execution/instruction-checking benchmark harnesses.

## Other repaired defects

- Invalid root `file:./...` dependency URLs prevented installation. Replaced them
  with standard package requirements and uv workspace sources; added `uv.lock`.
- MGSM referenced removed TSV files and ignored the requested split. It now uses
  the official versioned language configs and numeric answer field.
- CommonsenseQA/TruthfulQA used obsolete short IDs; TheoremQA decoded images unnecessarily. It now retains all 800 rows, records
  image presence, and sends only the text to the selected text-only models.
- MATH could silently substitute MATH-Hard or drop failed categories. It now fails
  explicitly and uses the final boxed answer. ARC numeric choices are mapped to
  letter labels so their prompts, labels, and parsers agree.
- Missing template fields could become empty strings. They now fail clearly.
  Belebele's simple template now includes the passage and answer choices.
- Every additional special token was treated as EOS. Termination now follows the
  selected model's generation configuration, with tokenizer EOS as fallback.
- Fixed base-tokenizer fallback, model layer probing before `_loaded`, repeated
  text/Unicode offset fallback, unsupported capture settings, and silent missing
  activation data.
- The writer previously overwrote existing runs and `_SUCCESS` did not validate
  tensor lengths. Existing output is rejected, structural checks run before
  completion, and total failure/empty datasets cannot be reported as successful.
- Fixed safety CLI duplicate arguments, split selection, configured greedy
  profiles, and token-budget enforcement. JBB defaults to goal-only collection;
  use an explicit artifact directory for jailbreak artifacts.
- Llama Guard used a hard-coded policy and invalid special-token strings. It now
  uses the selected Guard tokenizer's own chat template and strict safe/unsafe
  parsing. Multi-token labels do not receive misleading first-token probabilities.
- Restored documented `GenericTask` and `EvalPipeline` imports and refreshed saved
  evaluation labels on the existing `Run` object.

## GPU smoke test

Run from this checkout with an HF account authorized for the requested models:

```bash
uv sync --frozen --extra dev
uv run openact config-check configs/research.toml
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'

uv run openact collect --config configs/research.toml \
  --max-samples 3 --max-tokens 64 --output runs/gpu_smoke
uv run openact-validate runs/gpu_smoke --strict
uv run openact-eval eval runs/gpu_smoke --task math --evaluator parser
```

Repeat collection with each model and a **fresh** output directory:

```bash
uv run openact collect --config configs/research.toml \
  --model Qwen/Qwen2-7B-Instruct --max-samples 3 --max-tokens 64 \
  --output runs/gpu_smoke_qwen2
```

For the paper's Llama-2 safety collection, use `--task jbb --template zot` and a
new output directory. Safety now supports both `raw` and `zot`; the latter adds
the documented project CoT wrapper. Generation is greedy when the TOML is used.
HarmBench still requires prepared data; see the paper audit for fidelity gaps.

## Remaining limits

- Qwen2's short A100 40GB check passed with 15.26 GB peak CUDA allocation.
  Long responses, other checkpoints, and A100/H100 80GB remain untested.
  The teacher-forced pass has different memory use from generation; attention
  patterns require eager attention and can substantially increase memory use.
- `meta-llama/LlamaGuard-7b` returned HTTP 403 for tokenizer access in this
  environment. Its version remains provisional in the TOML. Obtain repository
  access and test it, or deliberately select another Guard version and record it.
  The repaired Guard path has unit coverage but no live Guard inference validation.
- No resume, async writer, or per-sample timeout implementation exists. The old
  README advertised these features incorrectly and has been corrected.
- The optional existing `test_08_smoke_run.py` tests skip without `runs/smoke`.
  They are not evidence of a completed GPU run.
- Parser and LLM-judge quality, large-run disk capacity, artifact attack coverage,
  and downstream GMM/ICL/FAR methodology are outside the execution checks above.

## Upstream references

The model-specific template behavior follows the
[HF chat-template documentation](https://huggingface.co/docs/transformers/chat_templating).
Final normalization placement is visible in the
[upstream Llama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).
MGSM language configurations come from its
[official dataset card](https://huggingface.co/datasets/juletxara/mgsm).
The provisional judge is the original
[Llama Guard model](https://huggingface.co/meta-llama/LlamaGuard-7b).
