# Lambda GPU workspace

## Host

- SSH alias: `gpu2`, public IP `132.145.194.205`, user `ubuntu`.
- Local key: `~/.ssh/lambda_gpu.pem`; agent forwarding supplies GitHub access.
- Tailscale is not used for this connection.
- GPU observed: NVIDIA A100-SXM4 **40GB**; driver `580.105.08`.
- Initial disk: 497 GB root volume, approximately 472 GB free.
- Remote checkout: `/home/ubuntu/openact`.

Connect using `ssh gpu2`, or select `gpu2` in Cursor Remote-SSH.

## Environment

### Verified on this instance

- Python 3.11.16; PyTorch 2.14.0 + CUDA 13.0; Transformers 5.17.0.
- CUDA and bf16 are available; a bf16 CUDA matrix multiplication passed.
- All 55 runtime/configuration tests passed on the server.
- Actual Qwen2-7B-Instruct collection passed: two 16-token samples, all 29 hidden
  entries, prompt-last, generation means, and pre/post final RMSNorm.
- Independent causal-prefix comparison: maximum relative L2 errors 1.97% and
  2.65%, within the recorded 3% bf16 tolerance. Peak CUDA allocation was 15.26 GB.
- Environment and raw check results: [lambda_environment.json](lambda_environment.json)
  and [lambda_qwen2_check.json](lambda_qwen2_check.json).
- Qwen2 weights are cached. Meta model weight requests return HTTP 401 until an
  authorized HF account is configured; server HF authentication is currently absent.
- No full dataset activation collection has been launched.

Create an isolated Python 3.11 environment from the committed lockfile:

```bash
cd ~/openact
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra dev --python 3.11
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'
uv run openact config-check configs/capability.toml
```

The lockfile selects CUDA runtime 13.0 on Linux. NVIDIA documents driver 580 or
newer as supporting CUDA 13.x minor-version compatibility; actual CUDA kernels
must also pass the workspace smoke checks. See
[NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Selected datasets

All selected data have been downloaded, rendered with `zot`, and written to
`/home/ubuntu/openact-data/prepared`. Prepared Parquet row counts were checked:

| Directory | Rows |
|---|---:|
| `math` | 5,000 |
| `mmlu` | 14,042 |
| `theoremqa` | 800 (53 image questions marked) |
| `belebele_en`, `belebele_de`, `belebele_zh`, `belebele_ar`, `belebele_es` | 900 each |

See [lambda_datasets.json](lambda_datasets.json). Each prepared directory also
contains the prompt template and pinned dataset sources in `prepared_manifest.json`.
To collect directly from these frozen prompts:

```bash
uv run openact collect --config configs/capability.toml \
  --task prepared --prepared-path /home/ubuntu/openact-data/prepared/mmlu \
  --output runs/mmlu_qwen2_prepared
```

That command launches a full collection; it has not been run during setup.

The [full capability configuration](../configs/capability.toml) defaults to Qwen2,
full MMLU, greedy zero-shot CoT, bf16 inference, and float32 all-layer storage with
both final RMSNorm sides. It has no sample limit. Safety is deferred.

```bash
# MMLU: all 14,042 test examples
uv run openact collect --config configs/capability.toml

# TheoremQA: all 800; 53 image questions retain text and image-presence metadata
uv run openact collect --config configs/capability.toml \
  --task theoremqa --output runs/theoremqa_qwen2

# BELEBELE: Qwen2, five languages, 900 examples per language
for lang in en de zh ar es; do
  uv run openact collect --config configs/capability.toml \
    --task belebele --language "$lang" --output "runs/belebele_qwen2_$lang" || break
done
```

These are full collection commands; environment preparation does not launch them.
Use a short GPU acceptance check first:

```bash
uv run python scripts/check_paper_collection.py --model qwen2 --output runs/qwen2_check
```

Meta checkpoints require a Hugging Face account with the relevant model access.
If authentication is needed, run `uv run hf auth login` interactively on the GPU.
Never commit tokens. Qwen2 is public.

All-layer token storage can exceed the current disk capacity over a full dataset.
Measure the actual bytes per sample in the smoke run, then plan output storage
before starting the full jobs; see [paper_collection.md](paper_collection.md).
