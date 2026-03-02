#!/usr/bin/env python
"""Download Llama Guard 3 models from HuggingFace.

Usage:
    python scripts/download_llamaguard.py --model-size 8B
    python scripts/download_llamaguard.py --model-size 1B
    python scripts/download_llamaguard.py --both

Requires HF_TOKEN env var (accept terms at huggingface.co first).
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_IDS = {
    "1B": "meta-llama/Llama-Guard-3-1B",
    "8B": "meta-llama/Llama-Guard-3-8B",
}


def download_llamaguard_model(
    model_size: str = "8B",
    output_dir: str = "storage/models",
) -> Path:
    if model_size not in MODEL_IDS:
        raise ValueError(f"Invalid model size.  Choose from: {list(MODEL_IDS.keys())}")

    model_id = MODEL_IDS[model_size]
    local_dir = Path(output_dir) / f"Llama-Guard-3-{model_size}"
    local_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    if not token:
        print("WARNING: No HF_TOKEN found.  You may need to accept terms at:")
        print(f"  https://huggingface.co/{model_id}")

    print(f"Downloading {model_id} -> {local_dir} ...")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=[
            "*.json",
            "*.bin",
            "*.safetensors",
            "tokenizer*",
            "*model*",
        ],
    )

    print(f"Done: {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download Llama Guard 3 models")
    parser.add_argument(
        "--model-size",
        type=str,
        default="8B",
        choices=["1B", "8B"],
        help="Model size to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Download both 1B and 8B models",
    )
    args = parser.parse_args()

    if args.both:
        for size in ["1B", "8B"]:
            try:
                download_llamaguard_model(size, args.output_dir)
            except Exception as e:
                print(f"Failed to download {size}: {e}")
    else:
        download_llamaguard_model(args.model_size, args.output_dir)


if __name__ == "__main__":
    main()