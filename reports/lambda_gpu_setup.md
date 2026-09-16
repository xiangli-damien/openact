# Lambda GPU workspace

The current experiment and measured readiness results are documented in
[matrix_readiness.md](matrix_readiness.md). The earlier Qwen-only pilot remains
available in [lambda_qwen2_check.json](lambda_qwen2_check.json).

## Connection and persistent paths

- SSH: `ssh gpu2` → `ubuntu@132.145.194.205`, direct public IP, no Tailscale.
- Local SSH identity: `~/.ssh/lambda_gpu.pem`; agent forwarding provides GitHub access.
- Actual GPU: **NVIDIA A100-SXM4 40GB**, driver 580.105.08.
- Persistent Lambda filesystem: `/lambda/nfs/dami`; `~/dami` points there.
- Active code: `/lambda/nfs/dami/openact`.
- Prepared data: `/lambda/nfs/dami/openact-data/prepared`.
- Results and logs: `/lambda/nfs/dami/openact/runs` and `logs`.
- Code, previous results, and prepared data were copied from the home directories
  and verified before using the persistent checkout. Home copies remain available.
- The active `.venv` links to `/home/ubuntu/openact/.venv`. The environment and HF
  weight cache remain on the instance disk; code and experiment outputs are persistent.
- The volume reports a virtual capacity of 8 EiB; this is **not a verified quota**.
  Compare the estimated storage in the matrix report with the actual Lambda allocation.

Connect in Cursor with Remote-SSH → `gpu2`, then open `~/dami/openact`.

## Environment

Python 3.11.16, PyTorch 2.14.0+cu130, Transformers 5.17.0, datasets 5.0.1,
Zarr 2.18.7, numcodecs 0.13.1. CUDA and bf16 matrix operations were verified.
The selected four generation models and Llama-Guard-3-8B are accessible and cached.
The original LlamaGuard-7b returned 403; the user selected Guard 3 instead.
No credentials are stored in the repository.

For a new instance, recreate the environment from the committed lockfile (the
existing instance already has a working environment):

```bash
cd ~/dami/openact
uv sync --frozen --extra dev --python 3.11
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'
.venv/bin/openact config-check configs/capability.toml
```

On a replacement host, first replace a stale `.venv` symlink if its old instance
path is absent. Authenticate interactively with `.venv/bin/hf auth login` when
needed; do not put tokens in Git or command-line arguments.

## Selected data

All selected inputs are prepared with `zot`, with pinned revisions and no empty
prompts or duplicate sample IDs. See [matrix_datasets.json](matrix_datasets.json).

| Folder | Rows | Models |
|---|---:|---|
| `math` | 5,000 | All four |
| `mmlu` | 14,042 | All four |
| `belebele_en`, `belebele_de`, `belebele_zh` | 900 each | All four |
| `theoremqa` | 800, including 53 image questions | All four, text input only |
| `wildjailbreak` | 50,050 Vanilla Harmful train prompts | Llama-2 only |

The previously prepared Arabic/Spanish files are retained but excluded from the
current three-language matrix. The configuration is
[experiment_matrix.toml](../configs/experiment_matrix.toml); the 20-example
research configuration is a separate demo setting.

## Commands

```bash
cd ~/dami/openact
git fetch origin
git merge --ff-only origin/main

# Three deterministic samples per cell; full 2,048-token budget.
.venv/bin/python scripts/run_collection_matrix.py --output runs/matrix_smoke_new

# Recheck existing activations and refresh evaluation labels without generation.
.venv/bin/python scripts/recheck_collection_matrix.py \
  runs/matrix_smoke_new/matrix_report.json \
  --output runs/matrix_smoke_new/matrix_report_rechecked.json

# Full matrix; this is a long, large collection and has NOT been started.
.venv/bin/python scripts/run_collection_matrix.py --full --output runs/matrix_full_new
```

Output folders must be fresh. There is no resume support. Use `tmux` for long
runs so an SSH disconnection does not end the process. Optional `--models qwen2`
(or `llama32`, `llama3`, `llama2`) limits the matrix to selected models.
The full command also evaluates capability responses and labels Safety with
Guard 3 after unloading generation models. GMM/ICL/FAR analysis remains external.
