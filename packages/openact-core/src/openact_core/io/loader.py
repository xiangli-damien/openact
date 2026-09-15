from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
if TYPE_CHECKING:
    from openact_core.io.run import Run


def _cast_label_array(raw: np.ndarray) -> np.ndarray:
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
        return int(self.hidden_states.shape[0])

    @property
    def label_names(self) -> List[str]:
        return list(self.labels.keys())

    def get_mask(self, label_name: str, value: Any) -> np.ndarray:
        arr = self.labels[label_name]
        if arr.dtype.kind == 'f':
            return np.array([not np.isnan(v) and v == value for v in arr], dtype=bool)
        return arr == value

    def select(self, mask: np.ndarray) -> 'LoadedData':
        return LoadedData(
            hidden_states=self.hidden_states[mask],
            labels={k: v[mask] for k, v in self.labels.items()},
            sample_indices=self.sample_indices[mask],
            reduction=self.reduction,
            layers=self.layers,
            model=self.model,
            dataset=self.dataset,
            meta=self.meta,
        )

    def split_by(self, label_name: str) -> Dict[Any, 'LoadedData']:
        arr = self.labels[label_name]
        values = [v for v in arr.tolist() if not (isinstance(v, float) and np.isnan(v))]
        unique = sorted(set(values), key=str)
        return {val: self.select(self.get_mask(label_name, val)) for val in unique}

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

        meta_json = json.dumps(
            {
                'reduction': self.reduction,
                'layers': self.layers,
                'model': self.model,
                'dataset': self.dataset,
                'label_names': list(self.labels.keys()),
                'meta': self.meta,
            },
            default=str,
        )
        arrays['__meta__'] = np.array([meta_json])
        np.savez_compressed(str(path), **arrays)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LoadedData':
        import json

        data = np.load(str(path), allow_pickle=True)
        info = json.loads(str(data['__meta__'][0]))
        labels: Dict[str, np.ndarray] = {}
        for name in info.get('label_names', []):
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
    def __init__(self, run: Union['Run', str, Path], only_valid: bool = True):
        from openact_core.io.run import Run

        self._run = run if isinstance(run, Run) else Run(run)
        self.only_valid = bool(only_valid)

    @property
    def run(self) -> 'Run':
        return self._run

    @property
    def available_labels(self) -> List[str]:
        return self._run.list_available_labels()

    def load(
        self,
        reduction: str = 'mean',
        layers: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        dtype: str = 'float32',
    ) -> LoadedData:
        indices = self._indices()
        hs = self._run.get_all_hidden_states(
            reduction=reduction,
            layers=layers,
            only_valid=self.only_valid,
            dtype=dtype,
        )
        return self._pack(hs, indices, labels, reduction, layers)

    def load_trajectories(
        self,
        n_points: int = 32,
        layers: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        dtype: str = 'float32',
    ) -> LoadedData:
        indices = self._indices()
        hs = self._trajectory(indices, n_points=n_points, layers=layers, dtype=dtype)
        return self._pack(hs, indices, labels, f'trajectory_{n_points}', layers)

    def load_range(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        reduction: str = 'mean',
        layers: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        dtype: str = 'float32',
    ) -> LoadedData:
        indices = self._indices()
        hs = self._range_reduce(
            indices,
            start=start,
            end=end,
            start_pct=start_pct,
            end_pct=end_pct,
            reduction=reduction,
            layers=layers,
            dtype=dtype,
        )
        return self._pack(hs, indices, labels, f'range_{reduction}', layers)

    def iter_per_token(
        self,
        layers: Optional[List[int]] = None,
        batch_size: int = 256,
        dtype: str = 'float32',
    ) -> Iterator[Tuple[int, np.ndarray]]:
        del batch_size
        indices = self._indices().tolist()
        hs_arr, sample_ptr = self._per_token_source()
        layer_idx = layers
        for i in indices:
            i = int(i)
            if i + 1 >= len(sample_ptr):
                yield i, np.empty((0, 0, 0), dtype=dtype)
                continue
            start = int(sample_ptr[i])
            end = int(sample_ptr[i + 1])
            token_hs = np.asarray(hs_arr[start:end])
            if layer_idx is not None and token_hs.size:
                token_hs = token_hs[:, layer_idx, :]
            yield i, token_hs.astype(dtype, copy=False)

    def iter_per_token_with_labels(
        self,
        layers: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        batch_size: int = 256,
        dtype: str = 'float32',
    ) -> Iterator[Tuple[int, np.ndarray, Dict[str, Any]]]:
        del batch_size
        label_names = self._resolve_labels(labels)
        indices = self._indices()
        label_arrays = self._read_labels(label_names)
        idx_to_pos = {int(idx): i for i, idx in enumerate(indices.tolist())}
        for idx, hs in self.iter_per_token(layers=layers, dtype=dtype):
            pos = idx_to_pos.get(int(idx))
            meta = {name: arr[pos] for name, arr in label_arrays.items()} if pos is not None else {}
            yield int(idx), hs, meta

    def load_labels_only(self, labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        indices = self._indices()
        label_names = self._resolve_labels(labels)
        return indices.copy(), self._read_labels(label_names)

    def _indices(self) -> np.ndarray:
        if self.only_valid:
            return self._run.get_valid_indices()
        return np.arange(len(self._run), dtype=np.int64)

    def _resolve_labels(self, labels: Optional[List[str]]) -> List[str]:
        if labels is None:
            return []
        if isinstance(labels, (list, tuple)):
            return [str(x) for x in labels]
        return [str(labels)]

    def _read_labels(self, label_names: List[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in label_names:
            raw = self._run.get_all_labels(name, only_valid=self.only_valid)
            out[name] = _cast_label_array(raw)
        return out

    def _pack(
        self,
        hidden_states: np.ndarray,
        indices: np.ndarray,
        labels: Optional[List[str]],
        reduction: str,
        layers: Optional[List[int]],
    ) -> LoadedData:
        label_names = self._resolve_labels(labels)
        label_arrays = self._read_labels(label_names)
        return LoadedData(
            hidden_states=hidden_states,
            labels=label_arrays,
            sample_indices=indices.copy(),
            reduction=reduction,
            layers=layers,
            model=self._run.manifest.model.name,
            dataset=self._run.manifest.dataset.name,
        )

    def _per_token_source(self):
        path = getattr(self._run, '_hs_per_token_path', None)
        if path is None or path not in self._run._zarr:
            raise ValueError('Per-token hidden states are not available in this run')
        hs_arr = self._run._zarr[path]
        sample_ptr = getattr(self._run, '_sample_ptr', None)
        if sample_ptr is None:
            sample_ptr = np.asarray(self._run._zarr['tokens/sample_ptr'][:]).astype(np.int64)
        return hs_arr, sample_ptr

    def _trajectory(
        self,
        indices: np.ndarray,
        n_points: int,
        layers: Optional[List[int]],
        dtype: str,
    ) -> np.ndarray:
        hs_arr, sample_ptr = self._per_token_source()
        n_layers_total = int(hs_arr.shape[1])
        hidden_dim = int(hs_arr.shape[2])
        layer_idx = layers if layers is not None else list(range(n_layers_total))
        out: List[np.ndarray] = []
        zero = np.zeros((n_points, len(layer_idx), hidden_dim), dtype=dtype)

        for i in indices.tolist():
            i = int(i)
            if i + 1 >= len(sample_ptr):
                out.append(zero)
                continue
            start = int(sample_ptr[i])
            end = int(sample_ptr[i + 1])
            n_tok = max(0, end - start)
            if n_tok <= 0:
                out.append(zero)
                continue
            token_hs = np.asarray(hs_arr[start:end])
            token_hs = token_hs[:, layer_idx, :]
            grid = np.linspace(0, n_tok - 1, num=n_points)
            pos = np.clip(np.round(grid).astype(np.int64), 0, n_tok - 1)
            out.append(token_hs[pos].astype(dtype, copy=False))

        if not out:
            return np.empty((0, n_points, len(layer_idx), hidden_dim), dtype=dtype)
        return np.stack(out, axis=0)

    def _range_reduce(
        self,
        indices: np.ndarray,
        start: Optional[int],
        end: Optional[int],
        start_pct: Optional[float],
        end_pct: Optional[float],
        reduction: str,
        layers: Optional[List[int]],
        dtype: str,
    ) -> np.ndarray:
        hs_arr, sample_ptr = self._per_token_source()
        n_layers_total = int(hs_arr.shape[1])
        hidden_dim = int(hs_arr.shape[2])
        layer_idx = layers if layers is not None else list(range(n_layers_total))
        zero = np.zeros((len(layer_idx), hidden_dim), dtype=dtype)

        out: List[np.ndarray] = []
        for i in indices.tolist():
            i = int(i)
            if i + 1 >= len(sample_ptr):
                out.append(zero)
                continue
            s = int(sample_ptr[i])
            e = int(sample_ptr[i + 1])
            n_tok = max(0, e - s)
            if n_tok <= 0:
                out.append(zero)
                continue

            a = int(start) if start is not None else None
            b = int(end) if end is not None else None
            if start_pct is not None:
                a = int(np.floor(float(start_pct) * n_tok))
            if end_pct is not None:
                b = int(np.ceil(float(end_pct) * n_tok))
            if a is None:
                a = 0
            if b is None:
                b = n_tok
            a = int(np.clip(a, 0, n_tok))
            b = int(np.clip(b, 0, n_tok))
            if b <= a:
                out.append(zero)
                continue

            token_hs = np.asarray(hs_arr[s + a : s + b])
            token_hs = token_hs[:, layer_idx, :]
            if token_hs.size == 0:
                out.append(zero)
                continue

            if reduction == 'mean':
                reduced = token_hs.mean(axis=0)
            elif reduction == 'first':
                reduced = token_hs[0]
            elif reduction == 'last':
                reduced = token_hs[-1]
            else:
                raise ValueError(f'Unknown reduction: {reduction!r}')
            out.append(reduced.astype(dtype, copy=False))

        if not out:
            return np.empty((0, len(layer_idx), hidden_dim), dtype=dtype)
        return np.stack(out, axis=0)
