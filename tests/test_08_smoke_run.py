"""08 — Smoke run: hidden states, alignment, loading modes.

Requires a completed run at ./runs/smoke/ (or any run you point it to).
Skip gracefully if no run exists.

Run:
    python -m pytest tests/test_08_smoke_run.py -v -s --tb=short
"""
from pathlib import Path
import numpy as np
import pytest

SMOKE_DIR = Path(__file__).resolve().parent.parent / "runs" / "smoke"


@pytest.fixture(scope="module")
def run():
    if not SMOKE_DIR.exists():
        pytest.skip(f"No smoke run at {SMOKE_DIR}")
    from openact_core import Run
    return Run(SMOKE_DIR)


def test_structure(run):
    m = run.manifest
    print(f"\n  Run: {run.run_dir}")
    print(f"  complete={run.is_complete}, samples={len(run)}, valid={run.n_valid}")
    print(f"  model={m.model.name}, n_layers={m.model.n_layers}, hidden_dim={m.model.hidden_dim}")
    print(f"  dataset={m.dataset.name}, split={m.dataset.split}")
    assert run.n_valid > 0


def test_sample_content(run):
    for idx in run.get_valid_indices()[:3].tolist():
        s = run[idx]
        print(f"\n  Sample {idx}: tokens={s.n_tokens}, finish={s.finish_reason}")
        print(f"    prompt:   {s.prompt_text[:120]}…")
        print(f"    response: {s.response_text[:120]}…")
        print(f"    GT:       {str(s.ground_truth)[:60]}")
        assert s.n_tokens > 0
        assert s.response_text


def test_per_token_hidden_states(run):
    s = run[int(run.get_valid_indices()[0])]
    hs = s.hidden_states
    print(f"\n  per-token hs: shape={hs.shape}, dtype={hs.dtype}")
    if hs.size == 0:
        pytest.skip("No per-token HS")
    T, L, H = hs.shape
    assert T == s.n_tokens, f"T={T} != n_tokens={s.n_tokens}"
    assert not np.all(hs == 0)
    print(f"  ✓ (T={T}, L={L}, H={H}), non-zero")


def test_mean_hidden_states(run):
    s = run[int(run.get_valid_indices()[0])]
    hs = s.mean_hidden_states
    print(f"\n  mean hs: shape={hs.shape}, dtype={hs.dtype}")
    if hs.size == 0:
        pytest.skip("No mean HS")
    assert hs.ndim == 2


def test_prompt_last(run):
    s = run[int(run.get_valid_indices()[0])]
    hs = s.prompt_last_hidden_states
    print(f"\n  prompt_last hs: shape={hs.shape}")
    if hs.size == 0:
        print("  (not available)")


def test_token_alignment(run):
    s = run[int(run.get_valid_indices()[0])]
    a = s.aligner
    issues = a.validate()
    print(f"\n  alignment: {a.n_tokens} tokens, text_len={len(a.text)}")
    print(f"  issues: {issues or 'none'}")
    for i in range(min(8, a.n_tokens)):
        sp = a.token_to_char_span(i)
        print(f"    [{i:3d}] {sp.start:4d}–{sp.end:4d}: {a.token_to_text(i)!r}")


def test_token_hs_correspondence(run):
    """Each token has text AND hidden state."""
    s = run[int(run.get_valid_indices()[0])]
    hs = s.hidden_states
    if hs.size == 0:
        pytest.skip("No per-token HS")
    print(f"\n  token → (text, ‖hs‖):")
    for i in range(min(5, s.n_tokens)):
        text = s.token_to_text(i)
        norm = np.linalg.norm(hs[i].astype(np.float32))
        print(f"    [{i}] text={text!r:20s} ‖hs‖={norm:.4f}")


def test_bulk_loading(run):
    for red in ("mean", "first", "last"):
        hs = run.get_all_hidden_states(reduction=red, only_valid=True)
        print(f"  reduction={red:5s}: shape={hs.shape}")
        assert hs.shape[0] == run.n_valid


def test_layer_selection(run):
    n = run.n_layers
    if n is None or n < 2:
        pytest.skip("Not enough layers")
    layers = [0, n - 1]
    hs = run.get_all_hidden_states(reduction="mean", layers=layers, only_valid=True)
    print(f"\n  layer selection {layers}: shape={hs.shape}")
    assert hs.shape[1] == 2


def test_dataloader(run):
    from openact_core import DataLoader
    loader = DataLoader(run, only_valid=True)
    data = loader.load(reduction="mean")
    print(f"\n  DataLoader: shape={data.hidden_states.shape}, labels={data.label_names}")
    assert data.n_samples == run.n_valid

    # per-token iterator
    count = 0
    for idx, hs in loader.iter_per_token():
        print(f"    per-token sample {idx}: {hs.shape}")
        count += 1
        if count >= 2:
            break


def test_labels(run):
    labels = run.list_available_labels()
    print(f"\n  Available labels: {labels}")
    idx = int(run.get_valid_indices()[0])
    for lbl in labels[:10]:
        val = run.get_label(idx, lbl)
        print(f"    {lbl:30s} = {str(val)[:60]}")


def test_validation(run):
    issues = run.validate()
    print(f"\n  Validation: {issues or '✓ no issues'}")