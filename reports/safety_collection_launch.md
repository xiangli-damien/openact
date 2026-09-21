# Balanced Llama-2 safety activation collection

User authorization (2026-09-21): collect 1,500 Guard-safe responses (including
refusals) and 1,500 Guard-unsafe responses. This is response safety, not factual
correctness. Generator: `meta-llama/Llama-2-7b-chat-hf`; judge:
`meta-llama/Llama-Guard-3-8B`. Both revisions are pinned in
`configs/safety_balanced.toml`.

## Procedure

1. Use the existing pinned WildJailbreak `vanilla_harmful` prepared prompts,
   with the recorded zero-shot CoT suffix. Remove exact duplicate prompts,
   deterministically shuffle, retain source IDs and source file hashes.
2. Screen 64 prompts first, then blocks of 256. HF batched greedy generation,
   bf16, SDPA, batch 8, maximum 2,048 new tokens. Left padding is masked; EOS
   is retained and following batch padding excluded. Batch metrics stop at
   each response's first EOS. Numerical results can differ from batch-size-1
   decoding; generated IDs and actual batch sizes are retained.
3. Unload the generator before loading Guard. Judge the exact rendered user
   prompt and complete decoded assistant response with Guard's own chat
   template. Store raw judge output/category/probabilities. Unknown labels
   do not count toward either quota.
4. Keep the first 1,500 valid responses of each class in the frozen random
   order. All screened text/IDs/judgments remain in screening records.
5. Reload the generator and teacher-force the saved IDs, without generating
   again. Save all layers, prompt last, all generated tokens including EOS,
   generation means, and final RMSNorm pre/post, using float32 storage of
   bf16 values. Ordinary OpenAct Runs and `safety`/`correctness` label files
   remain compatible with HSS. For harmful prompts, correctness means safe.
6. Write bounded 32-response activation shards to SSD, validate token IDs,
   arrays, RMS means and causal replay, then transfer to Dami with SHA256
   verification. Remove only the identical verified SSD copy.

Resume regenerates neither finished screening batches nor Guard judgments.
Published shards are verified before reuse. Interrupted local attempts are
retained; the launcher holds a process lock. A 50 GiB local disk reserve and
a 20,000-prompt screening cap prevent unbounded disk use or screening. If a
class is insufficient at the cap, status explicitly reports the shortfall;
it never fabricates or relabels examples. It does not automatically change
the generator, prompts, decoding, source subset or screening cap.

## Safety rate versus balanced dataset

Report natural screening counts (`safe`, `unsafe`, `unknown`) and safety rate
over successfully judged responses. This rate describes this prompt stream
and judge, not general model safety. The first 64 responses are a fixed pilot.
The eventual 1,500/1,500 selected dataset has a designed 50% class proportion
and cannot estimate natural safety. Prompt/behavior IDs must be grouped when
forming future train/test splits. Guard labels are predictions, not human
ground truth.

## Commands and outputs

Deploy source through GitHub before executing on Lambda:

```bash
.venv/bin/python -m pytest tests/test_18_safety_balanced.py -q
.venv/bin/python scripts/run_safety_balanced.py --config configs/safety_balanced.toml --max-blocks 1
# Resume after inspecting the pilot; run in tmux for persistence:
.venv/bin/python scripts/run_safety_balanced.py --config configs/safety_balanced.toml
```

Dami: `/lambda/nfs/dami/openact/runs/safety_balanced_20260921`.
Inspect `job_status.json`, `plan.json`, `screening/`, `published_shards.json`
and ordinary runs under `llama2/`. Final `selected_index.parquet` lists each
selected sample's run path, local row index, class, category and screen index.
`_SUCCESS` is written only when both verified quotas are met.

Sources: [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak),
[Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B).

## Pilot outcome and source adjustment

The frozen 64-question vanilla pilot yielded 64 safe, 0 unsafe, 0 unknown.
The user permits a nonidentical collection to obtain both classes. The next
run uses **existing official `adversarial_harmful` prompts**, with one variant
per underlying behavior and exact prompt deduplication. Both selected classes
come from this same source. No attacks are generated/optimized, no reference
completions are model inputs, and generator/judge/decoding stay the same.
The vanilla pilot remains a separate reference, not part of the balanced set.
The rate in the new run is an adversarial-benchmark rate, not a vanilla rate.

Prepare with `scripts/prepare_safety_benchmark.py --output
/lambda/nfs/dami/openact-data/prepared/wildjailbreak_adversarial_20260921` and use
`--config configs/safety_balanced_adversarial.toml` for the launcher. The new
output is `/lambda/nfs/dami/openact/runs/safety_balanced_adv_20260921`.
