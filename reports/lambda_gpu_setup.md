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
