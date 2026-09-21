# Safety HSS: prompt-last capture

The Safety analysis uses the last **model-input** token, including the chat
template, at every hidden layer. It predicts whether the subsequent response is
Guard-safe (including refusals); this is not a factual-correctness label.

## Resume collection

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u scripts/run_safety_balanced.py \
  --config configs/safety_balanced_adversarial.toml \
  --max-batch-size 4 --capture-mode prompt-last
```

`capture_policy.json` records the effective capture configuration and change
history separately from the frozen screening plan. The mode persists on resume.
Each shard's manifest, run config, and `_SHARD.json` records its actual scope.
Existing full-capture shards are retained and supply the same prompt-last arrays.
Do not rewrite their manifests or count diagnostic duplicate samples.

New shards contain:

- `hidden_states/prompt_last`: embedding plus all decoder output positions;
  Hugging Face's last entry is after final RMSNorm.
- `final_norm/pre/prompt_last` and `final_norm/post/prompt_last`.
- Exact prompt IDs, generated response IDs/text, response metrics, and Guard labels.
- No generation-token activations or generation-mean activations.

The forward replays the exact saved prompt and response IDs, with the same shape
as previous full captures, but extracts **only the prompt-last position**. Causal
attention prevents future answer tokens from affecting it. Saved tensors are
float32, preserving bf16 values. The original response still
has to be generated and judged, even after the safe quota is full. This change
reduces capture/storage cost, not the number of candidates needed for rare labels.
GPU testing found about 2% bf16 differences when switching to a shorter,
prompt-only forward. Keeping the original shape avoids confounding capture mode
with the label (the safe quota was filled before the switch). The GPU smoke test
requires exact equality with previously saved prompt-last values; CPU tests also
change future response token IDs and require the prompt state to stay identical.

## HSS reader

Use the existing HSS `DataSpec`/OpenAct reader. Both old and new shards are valid:

```python
from hss.data.spec import DataSpec
from hss.data.openact import prepare

spec = DataSpec(
    paths=["/lambda/nfs/dami/openact/runs/safety_balanced_adv_20260921/llama2/block_*/shard_*"],
    representation="prompt_last",
    final_norm="pre",  # Or "post" for the separate post-RMSNorm analysis.
    label_file="safety",
    label="is_correct",  # Stored alias: True = Guard-safe, False = Guard-unsafe.
    expected_model="meta-llama/Llama-2-7b-chat-hf",
)
states = prepare(spec, "/home/ubuntu/hss-safety-prompt-cache")
```

The reader uses prompt-last arrays directly and does not average response states.
`pre` replaces the final Hugging Face entry with the recorded pre-RMSNorm state;
earlier layers are unchanged. Do not use response length/entropy as inputs when
claiming a predictor available before generation. They remain useful diagnostic
covariates in a separately identified post-generation analysis.

The 1,500/1,500 target is a selected balance. Estimate the natural safety rate
from **all judged candidates**, not this subset. Split by the saved underlying
behavior ID when fitting/evaluating HSS to avoid prompt-variant leakage.
