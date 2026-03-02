"""
Export utilities for converting OpenAct runs to external formats.

Supports exporting to NumPy, HuggingFace datasets, and JSON alignment files.
"""
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
from openact_core.io.run import Run


# ============================================================================
# Type coercion utilities
# ============================================================================

def _ensure_run(run: Union[Run, str, Path]) -> Run:
    """Coerce a path or Run to a Run instance.
    
    Args:
        run: A Run instance or path to a run directory.
    
    Returns:
        Run instance.
    """
    if isinstance(run, Run):
        return run
    return Run(run)


def _ensure_path(path: Union[str, Path]) -> Path:
    """Coerce a string or Path to a Path instance."""
    return Path(path) if not isinstance(path, Path) else path


# ============================================================================
# Export functions
# ============================================================================

def export_to_numpy(
    run: Union[Run, str, Path],
    output_dir: Union[str, Path],
    reduction: str = "mean",
    layers: Optional[List[int]] = None,
    only_valid: bool = True,
    include_labels: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Export run hidden states and optional labels to NumPy files.

    Parameters
    ----------
    run : Run or path
        The run to export.
    output_dir : path
        Directory to write output files.
    reduction : str
        Token-level reduction: 'mean', 'first', or 'last'.
    layers : list of int, optional
        Layer indices to include (all layers if None).
    only_valid : bool
        If True, skip non-OK samples.
    include_labels : list of str, optional
        Label column names to export alongside hidden states.

    Returns
    -------
    dict
        Mapping of output name to file path.
    """
    run = _ensure_run(run)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    # Hidden states
    hs = run.get_all_hidden_states(
        reduction=reduction, layers=layers, only_valid=only_valid
    )
    hs_path = output_dir / "hidden_states.npy"
    np.save(hs_path, hs)
    outputs["hidden_states"] = hs_path

    # Optional labels
    if include_labels:
        labels_dict = {}
        for label_name in include_labels:
            arr = run.get_all_labels(
                label_name, only_valid=only_valid
            )
            # Convert object arrays to typed arrays for serialization
            # This avoids pickle issues while preserving type information
            if arr.dtype == object:
                # Check if array contains None values
                has_none = any(v is None for v in arr)
                non_none_vals = [v for v in arr if v is not None]
                
                if non_none_vals:
                    # Check if all non-None values are bool (and no None values)
                    if not has_none and all(isinstance(v, bool) for v in arr):
                        # Pure bool array: convert to numpy bool dtype
                        labels_dict[label_name] = np.array(arr, dtype=bool)
                    # Check if all non-None values are numeric (int or float, not bool)
                    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none_vals):
                        # Numeric array: convert to float, None becomes NaN
                        float_arr = np.array([float(v) if v is not None else np.nan for v in arr], dtype=float)
                        labels_dict[label_name] = float_arr
                    else:
                        # Mixed types, strings, or bool with None: keep as object (will use pickle)
                        labels_dict[label_name] = arr
                else:
                    # All None: keep as object array
                    labels_dict[label_name] = arr
            else:
                # Already typed array, use as-is
                labels_dict[label_name] = arr
        labels_path = output_dir / "labels.npz"
        np.savez(labels_path, **labels_dict)
        outputs["labels"] = labels_path

    # Metadata
    import json

    meta_path = output_dir / "metadata.json"

    stats_obj = run.stats
    stats_data = stats_obj.to_dict() if hasattr(stats_obj, "to_dict") else dict(stats_obj)

    # Determine effective layer / hidden_dim info
    n_layers = hs.shape[1] if hs.ndim >= 2 else None
    hidden_dim = hs.shape[2] if hs.ndim >= 3 else None

    metadata = {
        "source_run": str(run.run_dir),
        "n_samples": len(hs),
        "reduction": reduction,
        "layers": layers,
        "only_valid": only_valid,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "model": run.manifest.model.name,
        "dataset": run.manifest.dataset.name,
        "stats": stats_data,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    outputs["metadata"] = meta_path

    return outputs


def export_to_hf_dataset(
    run: Union[Run, str, Path],
    include_hidden_states: bool = False,
    reduction: str = "mean",
    layers: Optional[List[int]] = None,
    only_valid: bool = True,
):
    """Export run to a HuggingFace Dataset object."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    run = _ensure_run(run)

    df = run.to_dataframe()
    if only_valid:
        from openact_core.schema.status import SampleStatus

        df = df[df["status"] == SampleStatus.OK].reset_index(drop=True)

    if include_hidden_states:
        hs = run.get_all_hidden_states(
            reduction=reduction, layers=layers, only_valid=only_valid
        )
        df["hidden_states"] = [hs[i].tolist() for i in range(len(hs))]

    return Dataset.from_pandas(df)


def export_hidden_states_to_npy(
    run: Union[Run, str, Path],
    output_path: Union[str, Path],
    sample_indices: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    reduction: Optional[str] = None,
) -> Path:
    """Export per-sample hidden states to a single .npy file."""
    run = _ensure_run(run)
    output_path = _ensure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_indices is None:
        sample_indices = run.get_valid_indices().tolist()

    all_hs = []
    for idx in sample_indices:
        sample = run[idx]
        hs = sample.hidden_states

        if layers is not None:
            hs = hs[:, layers, :]

        if reduction == "mean":
            hs = hs.mean(axis=0, keepdims=True)
        elif reduction == "first":
            hs = hs[0:1]
        elif reduction == "last":
            hs = hs[-1:]

        all_hs.append(hs)

    if reduction is None:
        result = np.array(all_hs, dtype=object)
    else:
        result = np.concatenate(all_hs, axis=0)

    np.save(output_path, result)
    return output_path


def export_alignments(
    run: Union[Run, str, Path],
    output_path: Union[str, Path],
    sample_indices: Optional[List[int]] = None,
) -> Path:
    """Export token-character alignments to a JSON file."""
    import json

    run = _ensure_run(run)
    output_path = _ensure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_indices is None:
        sample_indices = run.get_valid_indices().tolist()

    alignments = []
    for idx in sample_indices:
        sample = run[idx]
        boundaries = sample.aligner.get_token_boundaries()
        alignments.append(
            {
                "sample_idx": idx,
                "sample_id": sample.sample_id,
                "response_text": sample.response_text,
                "n_tokens": sample.n_tokens,
                "tokens": [
                    {
                        "idx": i,
                        "char_start": start,
                        "char_end": end,
                        "text": text,
                    }
                    for i, (start, end, text) in enumerate(boundaries)
                ],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alignments, f, indent=2, ensure_ascii=False)

    return output_path