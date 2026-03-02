from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def _to_native(v: Any) -> Any:
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.str_):
        return str(v)
    return v


def _is_na(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    try:
        return pd.isna(v)
    except (TypeError, ValueError):
        return False


def _cast_array(raw: np.ndarray) -> np.ndarray:
    values = raw.tolist()
    non_none = [v for v in values if v is not None]
    if not non_none:
        return np.full(len(values), np.nan, dtype=np.float64)
    if all(isinstance(v, bool) for v in non_none):
        if any(v is None for v in values):
            out = np.full(len(values), np.nan, dtype=np.float64)
            for i, v in enumerate(values):
                if v is not None:
                    out[i] = float(v)
            return out
        return np.array(values, dtype=bool)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none):
        out = np.full(len(values), np.nan, dtype=np.float64)
        for i, v in enumerate(values):
            if v is not None:
                out[i] = float(v)
        return out
    if all(isinstance(v, str) for v in non_none):
        return np.array([v if v is not None else '' for v in values], dtype='U')
    return raw


@dataclass
class LoadedData:
    """Aligned container: hidden_states[i] corresponds to labels[k][i] and sample_indices[i]."""
    hidden_states: np.ndarray
    labels: Dict[str, np.ndarray]
    sample_indices: np.ndarray
    reduction: str = 'mean'
    layers: Optional[List[int]] = None
    model: Optional[str] = None
    dataset: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def n_layers(self) -> Optional[int]:
        if self.hidden_states.ndim >= 2:
            return self.hidden_states.shape[-2]
        return None

    @property
    def hidden_dim(self) -> Optional[int]:
        if self.hidden_states.ndim >= 2:
            return self.hidden_states.shape[-1]
        return None

    @property
    def label_names(self) -> List[str]:
        return list(self.labels.keys())

    def get_mask(self, label_name: str, value: Any) -> np.ndarray:
        arr = self.labels[label_name]
        if arr.dtype.kind == 'f':
            return np.array([not np.isnan(v) and v == value for v in arr], dtype=bool)
        return arr == value

    def select(self, mask: np.ndarray) -> LoadedData:
        return LoadedData(
            hidden_states=self.hidden_states[mask],
            labels={k: v[mask] for k, v in self.labels.items()},
            sample_indices=self.sample_indices[mask],
            reduction=self.reduction, layers=self.layers,
            model=self.model, dataset=self.dataset, meta=self.meta,
        )

    def split_by(self, label_name: str) -> Dict[Any, LoadedData]:
        arr = self.labels[label_name]
        unique = {v for v in arr.tolist() if not (isinstance(v, float) and np.isnan(v))}
        return {val: self.select(self.get_mask(label_name, val))
                for val in sorted(unique, key=str)}

    def save(self, path: Union[str, Path]) -> Path:
        import json
        path = Path(path)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {
            'hidden_states': self.hidden_states,
            'sample_indices': self.sample_indices,
        }
        for name, arr in self.labels.items():
            arrays[f'label__{name}'] = arr
        meta_json = json.dumps({
            'reduction': self.reduction, 'layers': self.layers,
            'model': self.model, 'dataset': self.dataset,
            'label_names': list(self.labels.keys()), 'meta': self.meta,
        }, default=str)
        arrays['__meta__'] = np.array([meta_json])
        np.savez_compressed(str(path), **arrays)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> LoadedData:
        import json
        data = np.load(str(path), allow_pickle=True)
        info = json.loads(str(data['__meta__'][0]))
        labels = {}
        for name in info['label_names']:
            key = f'label__{name}'
            if key in data:
                labels[name] = data[key]
        return cls(
            hidden_states=data['hidden_states'],
            labels=labels,
            sample_indices=data['sample_indices'],
            reduction=info.get('reduction', 'mean'),
            layers=info.get('layers'),
            model=info.get('model'),
            dataset=info.get('dataset'),
            meta=info.get('meta', {}),
        )

    def __repr__(self) -> str:
        lbl = ', '.join(f'{k}({v.dtype})' for k, v in self.labels.items())
        return f'LoadedData(n={self.n_samples}, shape={self.hidden_states.shape}, labels=[{lbl}])'


class DataLoader:
    """Unified loader for hidden states, labels, and metadata from an OpenAct run.

    Usage:
        loader = DataLoader(run)
        data = loader.load(reduction='mean', labels=['is_correct'])
        data = loader.load(reduction='prompt_last')
        data = loader.load_trajectories(n_points=32)
        data = loader.load_range(start_pct=0.5, end_pct=1.0, reduction='mean')
        for idx, hs in loader.iter_per_token():
            ...
    """

    def __init__(self, run: Union['Run', str, Path], only_valid: bool = True):
        from openact_core.io.run import Run
        self._run = run if isinstance(run, Run) else Run(run)
        self.only_valid = only_valid

    @property
    def run(self) -> 'Run':
        return self._run

    @property
    def available_labels(self) -> List[str]:
        return self._run.list_available_labels()

    # ── public API ──────────────────────────────────────────────

    def load(self, reduction: str = 'mean', layers: Optional[List[int]] = None,
             labels: Optional[List[str]] = None, dtype: str = 'float32') -> LoadedData:
        """Load hidden states with a per-sample reduction.

        reduction: 'mean' | 'prompt_last' | 'first' | 'last'
        Returns shape (N, L, D).
        """
        indices = self._indices()
        hs = self._load_reduced(reduction, layers, indices, dtype)
        return self._pack(hs, indices, labels, reduction, layers, dtype)

    def load_trajectories(self, n_points: int = 32, layers: Optional[List[int]] = None,
                          labels: Optional[List[str]] = None,
                          dtype: str = 'float32') -> LoadedData:
        """Load evenly-spaced trajectory points. Returns shape (N, T, L, D)."""
        indices = self._indices()
        hs = self._load_trajectory_data(indices, layers, n_points, dtype)
        return self._pack(hs, indices, labels, f'trajectory_{n_points}', layers, dtype)

    def load_range(self, start: Optional[int] = None, end: Optional[int] = None,
                   start_pct: Optional[float] = None, end_pct: Optional[float] = None,
                   reduction: str = 'mean', layers: Optional[List[int]] = None,
                   labels: Optional[List[str]] = None, dtype: str = 'float32') -> LoadedData:
        """Load hidden states from a token sub-range with reduction.

        Specify either (start, end) as token indices or (start_pct, end_pct) as fractions.
        Returns shape (N, L, D).
        """
        indices = self._indices()
        hs = self._load_range_data(indices, start, end, start_pct, end_pct,
                                    reduction, layers, dtype)
        tag = f'range_{reduction}'
        return self._pack(hs, indices, labels, tag, layers, dtype)

    def iter_per_token(self, layers: Optional[List[int]] = None,
                       batch_size: int = 256, dtype: str = 'float32'
                       ) -> Iterator[Tuple[int, np.ndarray]]:
        """Stream per-token hidden states sample by sample.

        Yields (sample_idx, hidden_states) where hidden_states is (T_i, L, D).
        Uses batched I/O internally for efficiency.
        """
        from openact_core.io.batch_reader import (
            RunSnapshot, TokenSelector, iter_batched_hidden_states,
        )
        indices = self._indices()
        snap = RunSnapshot(self._run)
        if snap.hs_path is None:
            raise ValueError('No per-token hidden states available in this run')
        for idx, _sel, hs in iter_batched_hidden_states(
            snap, indices.tolist(), TokenSelector.all_tokens,
            layers=layers, batch_size=batch_size, output_dtype=dtype,
        ):
            yield idx, hs

    def iter_per_token_with_labels(self, layers: Optional[List[int]] = None,
                                    labels: Optional[List[str]] = None,
                                    batch_size: int = 256, dtype: str = 'float32'
                                    ) -> Iterator[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """Stream per-token hidden states with per-sample labels.

        Yields (sample_idx, hidden_states, label_dict).
        """
        label_names = self._resolve_labels(labels)
        indices = self._indices()
        label_arrays = self._read_labels(label_names, indices)
        idx_to_pos = {int(idx): i for i, idx in enumerate(indices)}
        for idx, hs in self.iter_per_token(layers=layers, batch_size=batch_size, dtype=dtype):
            pos = idx_to_pos[idx]
            row = {name: arr[pos] for name, arr in label_arrays.items()}
            yield idx, hs, row

    def load_labels_only(self, labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load only labels and sample indices (no hidden states)."""
        indices = self._indices()
        label_names = self._resolve_labels(labels)
        label_arrays = self._read_labels(label_names, indices)
        return indices.copy(), label_arrays

    # ── internal: hidden state loading ──────────────────────────

    def _load_reduced(self, reduction: str, layers: Optional[List[int]],
                      indices: np.ndarray, dtype: str) -> np.ndarray:
        precomputed = self._try_precomputed(reduction, layers)
        if precomputed is not None:
            return precomputed.astype(dtype) if precomputed.dtype != np.dtype(dtype) else precomputed

        from openact_core.io.batch_reader import (
            RunSnapshot, TokenSelector, iter_batched_hidden_states,
        )
        snap = RunSnapshot(self._run)

        if snap.hs_path is None:
            raise ValueError(
                f'Cannot compute reduction={reduction!r}: no per-token hidden states. '
                f'Only precomputed reductions (mean, prompt_last) are available if stored.'
            )

        if reduction == 'mean':
            return self._batch_reduce(snap, indices, layers, dtype, 'mean',
                                       TokenSelector.all_tokens)
        elif reduction == 'first' or reduction == 'prompt_last':
            selector = lambda s, i: TokenSelector.token_range(s, i, start=0, end=1)
            return self._batch_reduce(snap, indices, layers, dtype, 'first', selector)
        elif reduction == 'last':
            selector = lambda s, i: TokenSelector.token_range(
                s, i, start=s.get_n_tokens(i) - 1, end=s.get_n_tokens(i))
            return self._batch_reduce(snap, indices, layers, dtype, 'first', selector)
        else:
            raise ValueError(f'Unknown reduction: {reduction!r}')

    def _batch_reduce(self, snap, indices, layers, dtype, reduce_op, selector_fn):
        from openact_core.io.batch_reader import iter_batched_hidden_states
        n_layers = len(layers) if layers else snap.n_layers
        zero = np.zeros((n_layers, snap.hidden_dim), dtype=dtype)
        results: Dict[int, np.ndarray] = {}
        for idx, _sel, hs in iter_batched_hidden_states(
            snap, indices.tolist(), selector_fn, layers=layers,
            batch_size=512, output_dtype=dtype,
        ):
            if len(hs) == 0:
                results[idx] = zero
            elif reduce_op == 'mean':
                results[idx] = hs.mean(axis=0)
            elif reduce_op == 'first':
                results[idx] = hs[0]
            elif reduce_op == 'last':
                results[idx] = hs[-1]
            else:
                results[idx] = hs.mean(axis=0)
        return np.stack([results[int(i)] for i in indices], axis=0)

    def _try_precomputed(self, reduction: str, layers: Optional[List[int]]) -> Optional[np.ndarray]:
        z = self._run._zarr
        path_map = {'mean': 'hidden_states/mean', 'prompt_last': 'hidden_states/prompt_last'}
        path = path_map.get(reduction)
        if path is None or path not in z:
            return None
        data = np.asarray(z[path][:])
        if self.only_valid:
            from openact_core.schema.status import SampleStatus
            mask = (self._run._df['status'] == SampleStatus.OK).values
            data = data[mask]
        if layers is not None:
            data = data[:, layers, :]
        return data

    def _load_trajectory_data(self, indices: np.ndarray, layers: Optional[List[int]],
                               n_points: int, dtype: str) -> np.ndarray:
        from openact_core.io.batch_reader import (
            RunSnapshot, TokenSelector, iter_batched_hidden_states,
        )
        snap = RunSnapshot(self._run)
        if snap.hs_path is None:
            raise ValueError('No per-token hidden states for trajectory loading')

        n_layers = len(layers) if layers else snap.n_layers
        selector = lambda s, i: TokenSelector.evenly_spaced(s, i, n_points)
        results: Dict[int, np.ndarray] = {}
        for idx, _sel, hs in iter_batched_hidden_states(
            snap, indices.tolist(), selector, layers=layers,
            batch_size=256, output_dtype=dtype,
        ):
            if len(hs) < n_points:
                padded = np.zeros((n_points, n_layers, snap.hidden_dim), dtype=dtype)
                padded[:len(hs)] = hs
                results[idx] = padded
            else:
                results[idx] = hs[:n_points]
        return np.stack([results[int(i)] for i in indices], axis=0)

    def _load_range_data(self, indices, start, end, start_pct, end_pct,
                          reduction, layers, dtype):
        from openact_core.io.batch_reader import (
            RunSnapshot, TokenSelector, iter_batched_hidden_states,
        )
        snap = RunSnapshot(self._run)
        if snap.hs_path is None:
            raise ValueError('No per-token hidden states for range loading')

        use_pct = start_pct is not None or end_pct is not None
        if use_pct:
            sp, ep = start_pct or 0.0, end_pct or 1.0
            selector = lambda s, i: TokenSelector.percentage_range(s, i, sp, ep)
        else:
            selector = lambda s, i: TokenSelector.token_range(s, i, start, end)

        reduce_op = reduction if reduction in ('mean', 'first', 'last') else 'mean'
        return self._batch_reduce(snap, indices, layers, dtype, reduce_op, selector)

    # ── internal: labels ────────────────────────────────────────

    def _resolve_labels(self, labels: Optional[List[str]]) -> List[str]:
        if labels is not None:
            return labels
        all_cols = self._run.list_available_labels()
        skip = {'status', 'sample_idx', 'index', 'sample_id', 'prompt_text',
                'response_text', 'full_text', 'finish_reason', 'ground_truth'}
        return [c for c in all_cols if c not in skip]

    def _read_labels(self, label_names: List[str], indices: np.ndarray) -> Dict[str, np.ndarray]:
        self._run._label_index._ensure_loaded()
        result: Dict[str, np.ndarray] = {}
        for name in label_names:
            result[name] = _cast_array(self._read_single_label(name, indices))
        return result

    def _read_single_label(self, name: str, indices: np.ndarray) -> np.ndarray:
        if name in self._run._df.columns:
            col = self._run._df[name].values
            raw = []
            for i in indices:
                v = col[i]
                raw.append(None if _is_na(v) else _to_native(v))
            return np.array(raw, dtype=object)

        indexed = self._run._label_index.get_column_indexed(name)
        if indexed is not None:
            return np.array([indexed.get(int(i)) for i in indices], dtype=object)

        return np.array([self._run.get_label(int(i), name) for i in indices], dtype=object)

    # ── internal: helpers ───────────────────────────────────────

    def _indices(self) -> np.ndarray:
        if self.only_valid:
            return self._run.get_valid_indices()
        return np.arange(len(self._run))

    def _pack(self, hs: np.ndarray, indices: np.ndarray,
              labels: Optional[List[str]], reduction: str,
              layers: Optional[List[int]], dtype: str) -> LoadedData:
        if hs.dtype != np.dtype(dtype):
            hs = hs.astype(dtype)
        label_names = self._resolve_labels(labels)
        label_arrays = self._read_labels(label_names, indices)
        return LoadedData(
            hidden_states=hs,
            labels=label_arrays,
            sample_indices=indices.copy(),
            reduction=reduction, layers=layers,
            model=self._run.manifest.model.name or None,
            dataset=self._run.manifest.dataset.name or None,
            meta={'run_dir': str(self._run.run_dir), 'only_valid': self.only_valid},
        )

    def __repr__(self) -> str:
        return (f"DataLoader(run='{self._run.run_dir.name}', "
                f"only_valid={self.only_valid}, labels={self.available_labels})")
