from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
from openact_core.io.run import Run
def _ensure_run(run: Union[Run, str, Path]) -> Run:
    if isinstance(run, Run):
        return run
    return Run(run)
def _ensure_path(path: Union[str, Path]) -> Path:
    return Path(path) if not isinstance(path, Path) else path
def export_to_numpy(
    run: Union[Run, str, Path],
    output_dir: Union[str, Path],
    reduction: str = "mean",
    layers: Optional[List[int]] = None,
    only_valid: bool = True,
    include_labels: Optional[List[str]] = None,
) -> Dict[str, Path]:
    run = _ensure_run(run)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    hs = run.get_all_hidden_states(
        reduction=reduction, layers=layers, only_valid=only_valid
    )
    hs_path = output_dir / "hidden_states.npy"
    np.save(hs_path, hs)
    outputs["hidden_states"] = hs_path
    if include_labels:
        labels_dict = {}
        for label_name in include_labels:
            arr = run.get_all_labels(
                label_name, only_valid=only_valid
            )
            if arr.dtype == object:
                has_none = any(v is None for v in arr)
                non_none_vals = [v for v in arr if v is not None]
                if non_none_vals:
                    if not has_none and all(isinstance(v, bool) for v in arr):
                        labels_dict[label_name] = np.array(arr, dtype=bool)
                    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none_vals):
                        float_arr = np.array([float(v) if v is not None else np.nan for v in arr], dtype=float)
                        labels_dict[label_name] = float_arr
                    else:
                        labels_dict[label_name] = arr
                else:
                    labels_dict[label_name] = arr
            else:
                labels_dict[label_name] = arr
        labels_path = output_dir / "labels.npz"
        np.savez(labels_path, **labels_dict)
        outputs["labels"] = labels_path
    import json
    meta_path = output_dir / "metadata.json"
    stats_obj = run.stats
    stats_data = stats_obj.to_dict() if hasattr(stats_obj, "to_dict") else dict(stats_obj)
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
