#!/usr/bin/env python3
"""
Smoke test: verify that a run's hidden states satisfy expected invariants.

Use after a collect run that saved all-layers per-token hidden states.
Checks:
  - hidden_states/per_token second dim == model n_layers (from manifest)
  - prompt_last shape matches (n_layers, hidden_dim)
  - generated_only_hidden_states length == n_tokens (no dropped first token)
  - Optionally: last-layer stats vs HF forward (not implemented here; manual check)

Usage:
  python scripts/smoke_test_hidden_states.py /path/to/run_dir
  python scripts/smoke_test_hidden_states.py /path/to/run_dir --max-samples 1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def smoke_test_run(run_dir: Path, max_samples: int = 3) -> Tuple[bool, List[str]]:
    """Run smoke checks on hidden states. Returns (all_passed, list of messages)."""
    from openact_core import Run
    from openact_core.schema.status import SampleStatus

    run_dir = Path(run_dir)
    if not run_dir.exists():
        return False, [f"Run directory not found: {run_dir}"]

    messages = []
    run = Run(run_dir, require_complete=False)

    # Expected dimensions from manifest (collect writes these)
    model = run.manifest.model
    n_layers = model.n_layers
    hidden_dim = model.hidden_dim
    if n_layers is None or hidden_dim is None:
        messages.append(
            "Manifest model has no n_layers/hidden_dim; cannot verify layer count."
        )
        n_layers = n_layers or 0
        hidden_dim = hidden_dim or 0

    # Check zarr has per_token
    if "hidden_states/per_token" not in run._zarr:
        messages.append("Run has no hidden_states/per_token; skip hidden-state checks.")
        return True, messages

    samples_checked = 0
    for sample in run.iter_valid():
        if samples_checked >= max_samples:
            break
        idx = sample.index
        try:
            n_tokens = sample.n_tokens
            hs = sample.hidden_states
            pl = sample.prompt_last_hidden_states
            gen_only = sample.generated_only_hidden_states
        except Exception as e:
            messages.append(f"Sample {idx}: error reading hidden states: {e}")
            continue

        # Invariant: per_token second dim == n_layers
        if len(hs.shape) != 3:
            messages.append(
                f"Sample {idx}: hidden_states.shape = {hs.shape}, expected (n_tokens, n_layers, hidden_dim)"
            )
        elif n_layers and hs.shape[1] != n_layers:
            messages.append(
                f"Sample {idx}: hidden_states.shape[1] = {hs.shape[1]} != manifest n_layers = {n_layers}"
            )
        elif hidden_dim and hs.shape[2] != hidden_dim:
            messages.append(
                f"Sample {idx}: hidden_states.shape[2] = {hs.shape[2]} != manifest hidden_dim = {hidden_dim}"
            )

        # prompt_last: (n_layers, hidden_dim)
        if len(pl.shape) != 2:
            messages.append(
                f"Sample {idx}: prompt_last_hidden_states.shape = {pl.shape}, expected (n_layers, hidden_dim)"
            )
        elif n_layers and pl.shape[0] != n_layers:
            messages.append(
                f"Sample {idx}: prompt_last shape[0] = {pl.shape[0]} != n_layers = {n_layers}"
            )
        elif hidden_dim and pl.shape[1] != hidden_dim:
            messages.append(
                f"Sample {idx}: prompt_last shape[1] = {pl.shape[1]} != hidden_dim = {hidden_dim}"
            )

        # generated_only: length == n_tokens (no dropped first token)
        if n_tokens > 0 and len(gen_only) != n_tokens:
            messages.append(
                f"Sample {idx}: generated_only_hidden_states length = {len(gen_only)} != n_tokens = {n_tokens} (first token may be dropped)"
            )

        # Consistency: hs should have same token count as n_tokens
        if hs.shape[0] != n_tokens:
            messages.append(
                f"Sample {idx}: hidden_states.shape[0] = {hs.shape[0]} != n_tokens = {n_tokens}"
            )

        messages.append(
            f"Sample {idx}: n_tokens={n_tokens}, hidden_states.shape={hs.shape}, "
            f"prompt_last.shape={pl.shape}, generated_only.len={len(gen_only)}"
        )
        samples_checked += 1

    if samples_checked == 0:
        messages.append("No valid samples with hidden states were checked.")
    else:
        messages.append(f"Checked {samples_checked} sample(s).")

    # Any line that looks like an error (contains "!=" or "expected" or "error" or "dropped")
    errors = [m for m in messages if "!=" in m or "expected" in m or "error" in m or "dropped" in m]
    all_passed = len(errors) == 0
    return all_passed, messages


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test hidden-state invariants for an OpenAct run."
    )
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Max number of valid samples to check (default: 3)",
    )
    args = parser.parse_args()

    passed, messages = smoke_test_run(args.run_dir, max_samples=args.max_samples)
    for m in messages:
        print(m)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
