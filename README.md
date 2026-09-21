

# OpenAct: LLM Activation Data Infrastructure

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**OpenAct** is an open-source infrastructure for collecting, storing, reading, and evaluating LLM activation data at scale. It provides a standardized data format and a modular toolchain that decouples GPU-heavy data collection from lightweight downstream analysis—so you can collect hidden states on a cluster and analyze them on a laptop.

```
 ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
 │ openact-     │      │ openact-     │      │ openact-     │
 │ collect      │─────▶│ core         │◀─────│ eval         │
 │              │      │              │      │              │
 │ GPU + torch  │      │ numpy/zarr   │      │ parsers +    │
 │ Model → Data │      │ Read / Align │      │ LLM judge    │
 └──────────────┘      └──────────────┘      └──────────────┘
       ▲                      ▲                      ▲
       │                      │                      │
   Collection             Your analysis          Evaluation
   (one-time)            (zero GPU dep)          (automated)
```

---

## Why OpenAct?

Mechanistic interpretability research typically involves ad-hoc scripts that tightly couple model inference, data storage, and analysis. This creates several pain points:

- **Reproducibility**: each project reinvents its own storage format, making it hard to share or reuse activation datasets.
- **Heavy dependencies**: reading a 50 MB hidden-state tensor shouldn't require installing PyTorch.
- **Text–token alignment**: mapping a substring like `"Answer: 42"` to its corresponding hidden states is surprisingly tricky, especially for byte-level tokenizers and CJK text.
- **Scale**: naive approaches (pickle per sample, HDF5 with GIL contention) don't scale to millions of tokens.

OpenAct solves these by defining a **standardized on-disk format** (Zarr + Parquet + JSON manifest) and providing three focused packages with a strict dependency direction: `collect → core ← eval`.

---

## Key Features

### Token–Text Hard Alignment
Every token is stored with its `(char_start, char_end)` offsets into the decoded response text. This means you can do:

```python
span = sample.find_text("the answer is")
hs = sample.get_span_hidden_states(span, layers=[-1], reduction="mean")
```

No tokenizer needed at read time. Works correctly for CJK, Arabic, multi-byte characters, and byte-level tokenizers (LLaMA, Qwen).

### Zero GPU for Reading
`openact-core` depends only on numpy, zarr, pandas, pyarrow, numcodecs, and pyyaml. You can load a 32-layer, 4096-dim hidden state dataset on a CPU-only machine, a CI server, or a Jupyter notebook without ever importing torch.

### Teacher-Forced Activation Collection
Generation chooses the response tokens. A separate teacher-forced forward pass over
prompt + response measures the raw states for those exact token positions. The
final normalization module is captured both before and after normalization.

### Synchronous Storage
The current Zarr writer writes each sample synchronously. `AsyncZarrWriter` is a
compatibility alias; `queue_size` does not enable background writes.

### Hierarchical Zarr Storage
Hidden states, attention patterns, and MLP activations are stored in a single Zarr group with lazy, chunked access. You can read one sample's hidden states without loading the entire dataset into memory.

### Complete Reproducibility
Every run writes a `manifest.json` capturing: model identifier + resolved revision, pinned dataset sources, actual chat template, prompt template (with SHA-256 hash), generation hyperparameters, capture configuration, environment (Python / torch / transformers versions, CUDA version, hostname, command line). A `_SUCCESS` marker is written only after structural integrity checks pass.

### Built-in Evaluation
`openact-eval` provides deterministic parser-based evaluation for 10+ benchmarks, LLM-as-judge evaluation (OpenAI / Anthropic backends), safety evaluation (refusal heuristics + LlamaGuard), and a unified pipeline that stores results as labels alongside the run.

---

## Installation

From this checkout (Python 3.11 is the tested environment):

```bash
uv sync --frozen --extra dev
```

Or install the local packages together with pip:

```bash
python -m pip install -e packages/openact-core -e packages/openact-collect -e packages/openact-eval -e .
```

`openact-core` can be installed by itself to read data without torch.
`openact-eval[llm]` adds API judge clients; these are optional.

See [the readiness audit](reports/readiness.md) for tested versions, known limits,
and the GPU smoke commands.

---

## Quick Start

### 1. Collect Activations

```bash
# Collect hidden states from GSM8K with Qwen2-7B-Instruct
openact collect \
    --model Qwen/Qwen2-7B-Instruct \
    --task gsm8k \
    --output runs/gsm8k_qwen \
    --max-samples 500 \
    --layers=-1,-2,-3          # last 3 layers only
```

Collection records per-sample errors and catches CUDA OOM. It rejects an existing
run directory: resume, per-sample timeouts, and async writing are not implemented.
A structurally valid run can contain failed samples; the CLI exits nonzero when
any sample fails. `_SUCCESS` is written after structural validation.

The supplied research configuration enables greedy zero-shot CoT, bf16 inference,
float32 storage at **every layer including embeddings**, prompt-last, every generated
token, and pre/post-final-norm capture. It starts with 20 MATH examples:

```bash
uv run openact config-check configs/research.toml
uv run openact collect --config configs/research.toml
```

See [collection against the supplied paper](reports/paper_collection.md) for model
dimensions, token conventions, remaining reproduction gaps, and the GPU acceptance
script for all four generation models.

Explicit CLI flags override TOML collection settings. The TOML's `analysis` tables
record your CPU GMM/ICL/alignment/split/smoothing/FAR settings for a downstream
fitter; they do not execute analysis. vLLM is not integrated.

Available tasks: `gsm8k`, `mgsm`, `mmlu`, `math`, `arc_challenge`, `commonsenseqa`, `belebele`, `theoremqa`, `truthfulqa`, `humaneval`, `ifeval`, `jbb`, `advbench`, `xstest`.

### 2. Read and Analyze (No GPU Needed)

```python
from openact_core import Run

run = Run("runs/gsm8k_qwen")
print(run)  # ✓ Run: gsm8k_qwen (487/500 valid)
print(run.stats)

# Access individual samples
sample = run[42]
print(sample.response_text[:200])
print(sample.hidden_states.shape)          # (T, L, H) e.g. (156, 3, 3584)
print(sample.mean_hidden_states.shape)     # (L, H)
print(sample.ground_truth)                 # "42"

# Token–text alignment: find a substring → get its hidden states
span = sample.find_text("Answer:")
if span:
    print(f"'Answer:' = tokens {span.start}:{span.end}")
    print(f"Verification: '{sample.span_to_text(span)}'")
    hs = sample.get_span_hidden_states(span, layers=[-1], reduction="mean")
    print(f"Hidden state vector: {hs.shape}")  # (H,)

# One-liner: text → hidden state
hs = sample.get_text_hidden_states("the answer is", layers=[-1])

# Bulk access for downstream ML
import numpy as np
X = run.get_all_hidden_states(reduction="mean", layers=[-1])  # (N, 1, H)
y = run.get_all_labels("is_correct")                           # (N,) bool
```

### Final normalization states

`hidden_states` keeps Hugging Face layer indexing: entry 0 is the embedding output,
and the last entry is the final norm output for the supported Llama/Qwen models.
The final block's residual output is available separately as `final_norm/pre`.

```python
before = sample.get_final_norm_states("pre")               # (T, H)
after = sample.get_final_norm_states("post")               # (T, H)
mean_before = sample.get_final_norm_states("pre", "mean")  # (H,)
prompt_before = sample.get_final_norm_states("pre", "prompt_last")
```

Values are stored without additional normalization. Token tensors include EOS or
other generated special tokens with zero-width text offsets. New manifests record
`custom.activation_extraction = "teacher_forced_forward"` and
`custom.token_alignment = "response_token"`; older runs are not rewritten.

### 3. Evaluate

```bash
# Auto-detect task and evaluator from manifest
openact-eval eval runs/gsm8k_qwen

# Explicit options
openact-eval eval runs/gsm8k_qwen --task gsm8k --evaluator parser --label correctness

# LLM-as-judge for open-ended tasks
openact-eval eval runs/truthfulqa_qwen --evaluator llm --llm-model gpt-4o-mini

# JSON output for scripting
openact-eval eval runs/gsm8k_qwen --json
```

Or programmatically:

```python
from openact_eval import EvalPipeline

pipeline = EvalPipeline("runs/gsm8k_qwen")
summary = pipeline.run_pipeline()
print(f"Accuracy: {summary['accuracy']:.1%}")
# Labels are automatically saved to runs/gsm8k_qwen/labels/correctness.parquet
```

---

## Data Format

Every OpenAct run is a self-contained directory:

```
runs/gsm8k_qwen/
├── manifest.json                   # Full reproducibility metadata
├── data.parquet                    # Per-sample metadata (prompt, response, metrics, ...)
├── _SUCCESS                        # Completion marker with validation results
├── labels/                         # Post-hoc labels (optional, added by eval or user)
│   ├── correctness.parquet
│   └── safety.parquet
└── tensors.zarr/                   # Activation tensors (Zarr directory store)
    ├── tokens/
    │   ├── ids                     # (N_total_tokens,) int32 — token IDs
    │   ├── offsets                 # (N_total_tokens, 2) int32 — char (start, end)
    │   └── sample_ptr              # (N_samples+1,) int64 — CSR-style pointers
    ├── sample_status               # (N_samples,) int8 — OK/ERROR/TIMEOUT/...
    └── hidden_states/
        ├── per_token               # (N_total_tokens, L, H) float16
        ├── mean                    # (N_samples, L, H) float32
        └── prompt_last             # (N_samples, L, H) float32
```

**Design choices:**

- **CSR-style token storage**: all tokens are concatenated into flat arrays; `sample_ptr[i]:sample_ptr[i+1]` gives sample `i`'s range. This avoids ragged arrays and enables efficient batch reads.
- **Char offsets with every token**: the `offsets` array stores `(char_start, char_end)` for each token relative to the decoded response text. This is the "hard alignment" that makes `find_text()` → `get_span_hidden_states()` work without a tokenizer.
- **Parquet for metadata, Zarr for tensors**: parquet is fast for tabular queries and filtering; Zarr provides chunked, compressed, lazy access to multi-dimensional arrays.
- **Labels as separate parquet files**: evaluation results, human annotations, or any post-hoc labels live in `labels/` and are joined by `sample_idx` at read time. This keeps the core data immutable.

---

## Package Details

### openact-core

The read-only data layer. No torch dependency.

| Module | Purpose |
|--------|---------|
| `io.Run` | Main entry point. Wraps zarr + parquet. Supports `__getitem__`, `iter_valid()`, `filter()`, bulk `get_all_*` methods. |
| `io.Sample` | Lazy proxy for one sample. Properties: `hidden_states`, `token_ids`, `response_text`, `aligner`, etc. |
| `io.BatchReader` | Efficient bulk hidden-state reads with merge-gap optimization. `HiddenStateLoader` for common patterns. |
| `alignment.TokenAligner` | Bidirectional token ↔ char mapping. `find_substring()`, `char_span_to_token_span()`, `visualize()`. |
| `tasks.parsers` | Answer extraction: `NumericParser`, `MCParser4/5`, `MathParser`, `CodeParser`, `RefusalParser`, etc. |
| `tasks.templates` | Prompt templates for all supported benchmarks, with multilingual prefix localization. |
| `export` | `export_to_numpy()`, `export_to_hf_dataset()`, `export_alignments()`. |

### openact-collect

The GPU-side collection engine.

| Module | Purpose |
|--------|---------|
| `engine.Collector` | Orchestrator. Handles plan iteration, per-sample error/OOM handling, parquet partitioning, and validation. |
| `engine.ModelManager` | HuggingFace model loading, chat template application, generation followed by a teacher-forced activation pass. |
| `engine.AsyncZarrWriter` | Synchronous writer (compatibility alias). Array resizing and sample pointer management. |
| `engine.OffsetCalculator` | Multi-strategy token→char offset computation: native offset mapping → verified prefix decoding, including split Unicode bytes. |
| `engine.OnlineMetricsProcessor` | Injects as a `LogitsProcessor` to compute perplexity, entropy, max-probability during generation with zero extra forward passes. |
| `extractors.HiddenStateExtractor` | Extracts per-token, mean, and prompt-last hidden states from `GenerationResult`. Supports layer selection and dtype casting. |
| `tasks.capability.*` | 11 benchmark tasks (GSM8K, MGSM, MMLU, MATH, ARC, CommonsenseQA, Belebele, TheoremQA, TruthfulQA, HumanEval, IFEval). |
| `tasks.safety.*` | 3 safety tasks (JBB, AdvBench, XSTest) with behavior × variant × profile plan generation and artifact (jailbreak prompt) loading. |

### openact-eval

Evaluation and labeling toolkit.

| Module | Purpose |
|--------|---------|
| `evaluators.ParserEvaluator` | Deterministic eval: parser extracts answer, matcher compares to ground truth. |
| `evaluators.LLMJudgeEvaluator` | LLM-as-judge with OpenAI/Anthropic backends, retry logic, structured output parsing. |
| `evaluators.RefusalHeuristicEvaluator` | Regex-based refusal detection for safety tasks. |
| `evaluators.LlamaGuardEvaluator` | Model-based safety classification using LlamaGuard. |
| `matchers` | `ExactMatcher`, `NumericMatcher` (relative tolerance), `MathMatcher` (expression evaluation), `ListMatcher`, `TypeAwareMatcher`. |
| `metrics` | `compute_metrics()`, `compare_evaluators()`, `MetricsSummary`. |
| `metrics_safety` | `compute_safety_metrics()`: ASR by category/method/profile, over-refusal rate, formatted report. |
| `pipeline.EvalPipeline` | One-call evaluation: auto-detect task → select evaluator → run → save labels → compute metrics. |

---

## Advanced Usage

### Batch Hidden-State Loading

For large-scale analysis, use the batch reader to avoid per-sample I/O overhead:

```python
from openact_core.io.batch_reader import HiddenStateLoader

run = Run("runs/gsm8k_qwen")
loader = HiddenStateLoader(run, layers=[-1], only_valid=True)

# Load all trajectories (variable-length per sample)
trajectories = loader.load_trajectories(batch_size=256)

# Fixed-length aligned trajectories (interpolated to 32 points)
aligned = loader.load_aligned_trajectories(n_points=32)
print(aligned.shape)  # (N_valid, 32, 1, H)

# Mean hidden states for a token range (e.g., last 20% of each response)
tail_mean = loader.load_range_mean(start_pct=0.8, end_pct=1.0)
print(tail_mean.shape)  # (N_valid, 1, H)
```

### Adding Custom Labels

Labels are stored as parquet files in `labels/` and are automatically accessible via `sample.get_label()`:

```python
import pandas as pd
from openact_core import Run

run = Run("runs/gsm8k_qwen")

# Your analysis produces labels
labels = pd.DataFrame({
    "sample_idx": [s.sample_idx for s in run.iter_valid()],
    "cluster_id": cluster_assignments,
    "confidence_score": confidence_scores,
})

labels.to_parquet(run.run_dir / "labels" / "my_analysis.parquet", index=False)

# Now accessible on every sample
sample = run[0]
print(sample.get_label("cluster_id"))

# Or bulk access
all_clusters = run.get_all_labels("cluster_id", only_valid=True)
```

### Safety Evaluation

```bash
# Collect safety data with multi-temperature sampling
openact collect \
    --model Qwen/Qwen2-7B-Instruct \
    --task jbb \
    --output runs/jbb_qwen \
    --profiles "greedy:temp=0.0 warm:temp=0.7,n_gen=2 hot:temp=1.0,n_gen=2" \
    --artifact-dir /path/to/attack_artifacts \
    --safety-split all

# Evaluate
openact-eval eval runs/jbb_qwen --task jbb
```

```python
from openact_eval import RefusalHeuristicEvaluator, compute_safety_metrics, format_safety_report

evaluator = RefusalHeuristicEvaluator()
result = evaluator.evaluate(run)
metrics = compute_safety_metrics(result)
print(format_safety_report(metrics))
# ══════════════════════════════════════════════════════════
#   SAFETY EVALUATION REPORT
# ══════════════════════════════════════════════════════════
#   Refusal rate:        87.5%
#   Attack success rate: 12.5%
#   ASR by attack method:
#     GCG                   23.1%
#     PAIR                  15.4%
#     GOAL                   3.8%
```

### Custom Tasks

```python
from openact_collect import GenericTask, Collector, ModelManager
from openact_collect.schema import CaptureSpec, GenerationSpec

# Define your data
data = [
    {"prompt": "Translate to French: Hello", "answer": "Bonjour"},
    {"prompt": "Translate to French: Goodbye", "answer": "Au revoir"},
]

task = GenericTask(data=data, prompt_key="prompt", answer_key="answer")
model = ModelManager("Qwen/Qwen2-7B-Instruct")

collector = Collector(
    model_manager=model,
    task=task,
    output_dir="runs/custom_translation",
    capture_spec=CaptureSpec(hidden_states_layers=[-1]),
    generation_spec=GenerationSpec(max_new_tokens=256, temperature=0.0),
)
stats = collector.run()
```

### Exporting Data

```python
from openact_core.export import export_to_numpy, export_to_hf_dataset

# Export to numpy arrays
paths = export_to_numpy(
    run="runs/gsm8k_qwen",
    output_dir="exports/gsm8k",
    reduction="mean",
    layers=[-1],
    include_labels=["is_correct"],
)
# Creates: hidden_states.npy, labels.npz, metadata.json

# Export to HuggingFace Dataset
dataset = export_to_hf_dataset("runs/gsm8k_qwen", include_hidden_states=True)
dataset.push_to_hub("username/gsm8k-qwen-activations")
```

## Supported Tasks

| Task | Type | Source | Parser | Evaluator |
|------|------|--------|--------|-----------|
| GSM8K | Math | `openai/gsm8k` | Numeric | Parser |
| MGSM | Math (multilingual) | `juletxara/mgsm` | Numeric | Parser |
| MATH | Math (competition) | `EleutherAI/hendrycks_math` | Math expression | Parser |
| MMLU | Knowledge (57 subjects) | `cais/mmlu` | MC-4 | Parser |
| ARC-Challenge | Reasoning | `allenai/ai2_arc` | MC-5 | Parser |
| CommonsenseQA | Reasoning | `tau/commonsense_qa` | MC-5 | Parser |
| Belebele | Reading comprehension | `facebook/belebele` | MC-4 | Parser |
| TheoremQA | Math/Science | `TIGER-Lab/TheoremQA` | Type-aware | Parser |
| TruthfulQA | Truthfulness | `truthfulqa/truthful_qa` | Freeform | LLM Judge |
| HumanEval | Code | `openai/openai_humaneval` | Code | LLM Judge |
| IFEval | Instruction following | `google/IFEval` | Freeform | LLM Judge |
| JailbreakBench | Safety | `JailbreakBench/JBB-Behaviors` | Refusal | Safety |
| AdvBench | Safety | `S3IC/advbench` | Refusal | Safety |
| XSTest | Safety (over-refusal) | `Paul/XSTest` | Refusal | Safety |

---

## Project Structure

```
openact/
├── pyproject.toml                          # Top-level meta-package
├── README.md
└── packages/
    ├── openact-core/                       # Data layer (no torch)
    │   └── src/openact_core/
    │       ├── io/                         # Run, Sample, BatchReader
    │       ├── alignment/                  # TokenAligner
    │       ├── schema/                     # Manifest, SampleStatus
    │       ├── tasks/                      # Templates, Parsers, Descriptors
    │       ├── export/                     # numpy, HuggingFace converters
    │       └── cli/                        # openact-validate CLI
    ├── openact-collect/                    # Collection engine (requires torch)
    │   └── src/openact_collect/
    │       ├── engine/                     # Collector, ModelManager, AsyncWriter
    │       ├── extractors/                 # HiddenStateExtractor
    │       ├── schema/                     # CaptureSpec, GenerationSpec, SafetySpec
    │       └── tasks/                      # Task implementations
    │           ├── capability/             # 11 benchmarks
    │           └── safety/                 # 3 safety datasets
    └── openact-eval/                       # Evaluation toolkit
        └── src/openact_eval/
            ├── evaluators/                 # Parser, LLM Judge, Safety
            ├── matchers/                   # Exact, Numeric, Math, List
            ├── metrics.py                  # Accuracy, comparison
            ├── metrics_safety.py           # ASR, over-refusal
            └── pipeline.py                 # EvalPipeline
```

---

## Development

```bash
uv sync --frozen --extra dev
uv run pytest tests
uv run python scripts/audit_data_and_prompts.py
uv build --all-packages
```

The original dataset tests require network access or a populated Hugging Face
cache. `tests/test_10_runtime.py` and `tests/test_11_config_and_integrity.py` run
offline with tiny real models and cover activation values, storage, configuration,
error handling, and evaluation. The optional tests in `test_08_smoke_run.py` read
an existing `runs/smoke` directory.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
