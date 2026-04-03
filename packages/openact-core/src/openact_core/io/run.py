from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from openact_core.io.run_stats import RunStats
from openact_core.io.sample import Sample
from openact_core.schema.manifest import Manifest
from openact_core.schema.status import SampleStatus


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if np.isnan(f) else f
    if isinstance(value, np.str_):
        s = str(value)
        return None if s == '' else s
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


class _LabelStore:
    def __init__(self, labels_dir: Path):
        self._labels_dir = Path(labels_dir)
        self._loaded = False
        self._frames: Dict[str, pd.DataFrame] = {}
        self._file_index_col: Dict[str, str] = {}
        self._label_to_file: Dict[str, str] = {}

    def invalidate(self) -> None:
        self._loaded = False
        self._frames.clear()
        self._file_index_col.clear()
        self._label_to_file.clear()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._labels_dir.exists():
            return
        for path in sorted(self._labels_dir.glob('*.parquet')):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if df.empty:
                continue
            idx_col = 'sample_idx' if 'sample_idx' in df.columns else ('index' if 'index' in df.columns else None)
            if idx_col is None:
                continue
            df = df.copy()
            df[idx_col] = df[idx_col].astype(int)
            df = df.set_index(idx_col, drop=False)
            stem = path.stem
            self._frames[stem] = df
            self._file_index_col[stem] = idx_col
            for col in df.columns:
                if col in (idx_col,):
                    continue
                self._label_to_file[col] = stem

    def list_labels(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._label_to_file.keys())

    def get(self, sample_idx: int, label_name: str) -> Any:
        self._ensure_loaded()
        stem = self._label_to_file.get(label_name)
        if stem is None:
            return None
        df = self._frames.get(stem)
        if df is None:
            return None
        if int(sample_idx) not in df.index:
            return None
        try:
            value = df.at[int(sample_idx), label_name]
        except Exception:
            return None
        return _to_python(value)

    def get_column(self, label_name: str) -> Optional[pd.Series]:
        self._ensure_loaded()
        stem = self._label_to_file.get(label_name)
        if stem is None:
            return None
        df = self._frames.get(stem)
        if df is None or label_name not in df.columns:
            return None
        return df[label_name]

    def get_column_indexed(self, label_name: str) -> Optional[Dict[int, Any]]:
        series = self.get_column(label_name)
        if series is None:
            return None
        out: Dict[int, Any] = {}
        for idx, v in series.items():
            out[int(idx)] = _to_python(v)
        return out

    def frames(self) -> Dict[str, pd.DataFrame]:
        self._ensure_loaded()
        return dict(self._frames)


class Run:
    def __init__(self, run_dir: Union[str, Path], require_complete: bool = False):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f'Run directory not found: {self.run_dir}')

        self._success_path = self.run_dir / '_SUCCESS'
        self._is_complete = self._success_path.exists()
        self._completion_info: Optional[Dict[str, Any]] = None
        if self._is_complete:
            try:
                self._completion_info = json.loads(self._success_path.read_text(encoding='utf-8'))
            except Exception:
                self._completion_info = None

        if require_complete and not self._is_complete:
            raise ValueError(f'Run is not marked complete: {self.run_dir}')

        manifest_path = self.run_dir / 'manifest.json'
        if manifest_path.exists():
            self.manifest = Manifest.load(manifest_path)
        else:
            self.manifest = Manifest()

        zarr_path = self.run_dir / 'tensors.zarr'
        if not zarr_path.exists():
            raise FileNotFoundError(f'Missing tensors.zarr in: {self.run_dir}')
        self._zarr = self._open_zarr(zarr_path)

        parquet_path = self.run_dir / 'data.parquet'
        if parquet_path.exists():
            self._df = self._read_parquet(parquet_path)
        else:
            legacy_path = self.run_dir / 'parquet' / 'records'
            if legacy_path.exists():
                self._df = self._read_parquet(legacy_path)
            else:
                raise FileNotFoundError(f'Missing data.parquet in: {self.run_dir}')

        self._labels = _LabelStore(self.run_dir / 'labels')
        self._stats: Optional[RunStats] = None

        self._sample_ptr = self._try_read_array('tokens/sample_ptr')
        self._hs_per_token_path = self._detect_hidden_states_path()

    @staticmethod
    def _open_zarr(zarr_path: Path):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError('zarr is required to read tensors.zarr') from exc
        return zarr.open_group(str(zarr_path), mode='r')

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        if path.is_dir():
            try:
                import pyarrow.dataset as ds
            except ImportError as exc:
                raise ImportError('pyarrow is required to read parquet datasets') from exc
            table = ds.dataset(str(path), format='parquet').to_table()
            return table.to_pandas()
        return pd.read_parquet(path)

    def _try_read_array(self, path: str) -> Optional[np.ndarray]:
        if path not in self._zarr:
            return None
        return np.asarray(self._zarr[path][:]).astype(np.int64)

    def _detect_hidden_states_path(self) -> Optional[str]:
        for path in ('hidden_states/per_token', 'response/hidden_states/data'):
            if path in self._zarr:
                return path
        return None

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def completion_info(self) -> Optional[Dict[str, Any]]:
        return self._completion_info

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, idx: int) -> bool:
        return 0 <= int(idx) < len(self)

    def __getitem__(self, idx: int) -> Sample:
        i = int(idx)
        if i < 0:
            i = len(self) + i
        if i < 0 or i >= len(self):
            raise IndexError(f'Index {idx} out of range [0, {len(self)})')
        return Sample(self, i)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def iter_valid(self) -> Iterator[Sample]:
        for i in self.get_valid_indices().tolist():
            yield self[int(i)]

    def iter_by_status(self, status: SampleStatus) -> Iterator[Sample]:
        for i in self.get_indices_by_status(status).tolist():
            yield self[int(i)]

    def filter(self, predicate: Callable[[Sample], bool]) -> List[Sample]:
        return [s for s in self.iter_valid() if predicate(s)]

    def filter_by_label(self, label_name: str, value: Any) -> List[Sample]:
        return self.filter(lambda s: s.get_label(label_name) == value)

    def get_indices_by_status(self, status: Union[SampleStatus, int]) -> np.ndarray:
        if 'status' not in self._df.columns:
            return np.array([], dtype=np.int64)
        code = int(status.value) if isinstance(status, SampleStatus) else int(status)
        col = self._df['status'].to_numpy()
        return np.where(col == code)[0].astype(np.int64)

    def get_valid_indices(self) -> np.ndarray:
        return self.get_indices_by_status(SampleStatus.OK)

    @property
    def n_total(self) -> int:
        return len(self)

    @property
    def n_valid(self) -> int:
        if 'status' not in self._df.columns:
            return len(self)
        return int((self._df['status'] == int(SampleStatus.OK)).sum())

    @property
    def n_layers(self) -> Optional[int]:
        for path in ('hidden_states/per_token', 'hidden_states/mean', 'response/hidden_states/data'):
            if path in self._zarr:
                return int(self._zarr[path].shape[1])
        return self.manifest.model.n_layers

    @property
    def hidden_dim(self) -> Optional[int]:
        for path in ('hidden_states/per_token', 'hidden_states/mean', 'response/hidden_states/data'):
            if path in self._zarr:
                return int(self._zarr[path].shape[2])
        return self.manifest.model.hidden_dim

    def list_available_labels(self) -> List[str]:
        cols = set(self._df.columns)
        cols |= set(self._labels.list_labels())
        internal = {'sample_idx', 'index', 'sample_id', 'status'}
        return sorted([c for c in cols if c not in internal])

    def refresh_labels(self) -> None:
        self._labels.invalidate()

    def get_label(self, sample_idx: int, label_name: str) -> Any:
        if label_name in self._df.columns:
            try:
                value = self._df.iloc[int(sample_idx)][label_name]
            except Exception:
                return None
            return _to_python(value)
        return self._labels.get(int(sample_idx), label_name)

    def get_all_labels(self, label_name: str, only_valid: bool = True) -> np.ndarray:
        indices = self.get_valid_indices() if only_valid else np.arange(len(self), dtype=np.int64)
        if label_name in self._df.columns:
            col = self._df[label_name].to_numpy(dtype=object)
            out = [_to_python(col[int(i)]) for i in indices.tolist()]
            return np.array(out, dtype=object)
        indexed = self._labels.get_column_indexed(label_name)
        if indexed is not None:
            out = [indexed.get(int(i)) for i in indices.tolist()]
            return np.array(out, dtype=object)
        out = [self.get_label(int(i), label_name) for i in indices.tolist()]
        return np.array(out, dtype=object)

    def _read_rows(self, arr, indices: Sequence[int]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for i in indices:
            rows.append(np.asarray(arr[int(i)]))
        if not rows:
            shape = (0,) + tuple(arr.shape[1:])
            return np.empty(shape, dtype=arr.dtype)
        return np.stack(rows, axis=0)

    def get_all_mean_hidden_states(
        self,
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
    ) -> np.ndarray:
        if 'hidden_states/mean' in self._zarr:
            indices = self.get_valid_indices() if only_valid else np.arange(len(self), dtype=np.int64)
            data = self._read_rows(self._zarr['hidden_states/mean'], indices.tolist())
            if layers is not None:
                data = data[:, layers, :]
            return data
        return self.get_all_hidden_states(reduction='mean', layers=layers, only_valid=only_valid)

    def get_all_prompt_last_hidden_states(
        self,
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
    ) -> Optional[np.ndarray]:
        if 'hidden_states/prompt_last' not in self._zarr:
            return None
        indices = self.get_valid_indices() if only_valid else np.arange(len(self), dtype=np.int64)
        data = self._read_rows(self._zarr['hidden_states/prompt_last'], indices.tolist())
        if layers is not None:
            data = data[:, layers, :]
        return data

    def get_all_hidden_states(
        self,
        reduction: str = 'mean',
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
        dtype: str = 'float32',
    ) -> np.ndarray:
        indices = self.get_valid_indices() if only_valid else np.arange(len(self), dtype=np.int64)

        if reduction == 'prompt_last':
            data = self.get_all_prompt_last_hidden_states(layers=layers, only_valid=only_valid)
            if data is None:
                raise ValueError('prompt_last hidden states are not available in this run')
            return data.astype(dtype, copy=False)

        if reduction == 'mean' and 'hidden_states/mean' in self._zarr:
            data = self._read_rows(self._zarr['hidden_states/mean'], indices.tolist())
            if layers is not None:
                data = data[:, layers, :]
            return data.astype(dtype, copy=False)

        n_layers = len(layers) if layers is not None else int(self.n_layers or 0)
        hidden_dim = int(self.hidden_dim or 0)
        zeros = np.zeros((n_layers, hidden_dim), dtype=dtype)

        if self._hs_per_token_path is not None and self._hs_per_token_path in self._zarr and self._sample_ptr is not None:
            hs_arr = self._zarr[self._hs_per_token_path]
            layer_idx = layers if layers is not None else list(range(int(hs_arr.shape[1])))
            n_layers = len(layer_idx)
            zeros = np.zeros((n_layers, int(hs_arr.shape[2])), dtype=dtype)

            out: List[np.ndarray] = []
            for i in indices.tolist():
                i = int(i)
                if i + 1 >= len(self._sample_ptr):
                    out.append(zeros)
                    continue
                start = int(self._sample_ptr[i])
                end = int(self._sample_ptr[i + 1])
                if end <= start:
                    out.append(zeros)
                    continue
                token_hs = np.asarray(hs_arr[start:end])
                if token_hs.size == 0:
                    out.append(zeros)
                    continue
                token_hs = token_hs[:, layer_idx, :]
                if reduction == 'mean':
                    reduced = token_hs.mean(axis=0)
                elif reduction == 'first':
                    reduced = token_hs[0]
                elif reduction == 'last':
                    reduced = token_hs[-1]
                else:
                    raise ValueError(f'Unknown reduction: {reduction!r}')
                out.append(reduced.astype(dtype, copy=False))
            return np.stack(out, axis=0) if out else np.empty((0, n_layers, int(hs_arr.shape[2])), dtype=dtype)

        out = []
        for i in indices.tolist():
            hs = self[int(i)].hidden_states
            if hs is None or len(hs) == 0:
                out.append(zeros)
                continue
            if layers is not None:
                hs = hs[:, layers, :]
            if reduction == 'mean':
                reduced = hs.mean(axis=0)
            elif reduction == 'first':
                reduced = hs[0]
            elif reduction == 'last':
                reduced = hs[-1]
            else:
                raise ValueError(f'Unknown reduction: {reduction!r}')
            out.append(reduced.astype(dtype, copy=False))
        return np.stack(out, axis=0) if out else np.empty((0, n_layers, hidden_dim), dtype=dtype)

    def get_all_generation_metrics(self, only_valid: bool = True) -> pd.DataFrame:
        cols = ['sample_idx', 'perplexity', 'entropy', 'max_probability']
        available = [c for c in cols if c in self._df.columns]
        df = self._df[available].copy() if available else pd.DataFrame(index=np.arange(len(self)))
        if only_valid and 'status' in self._df.columns:
            df = df[self._df['status'] == int(SampleStatus.OK)]
        return df

    def get_all_token_ids(self, only_valid: bool = True) -> List[np.ndarray]:
        iterator = self.iter_valid() if only_valid else self
        return [s.token_ids for s in iterator]

    def to_dataframe(self, include_labels: bool = True) -> pd.DataFrame:
        df = self._df.copy()
        if not include_labels:
            return df
        base_key = 'sample_idx' if 'sample_idx' in df.columns else ('index' if 'index' in df.columns else None)
        if base_key is None:
            return df
        for stem, ldf in self._labels.frames().items():
            idx_col = self._labels._file_index_col.get(stem)
            if idx_col is None or idx_col not in ldf.columns:
                continue
            cols = [c for c in ldf.columns if c != idx_col]
            if not cols:
                continue
            ldf_merge = ldf[[idx_col] + cols].reset_index(drop=True)
            if idx_col == base_key:
                df = df.merge(ldf_merge, how='left', on=base_key)
            else:
                df = df.merge(ldf_merge, how='left', left_on=base_key, right_on=idx_col)
                if idx_col in df.columns:
                    df = df.drop(columns=[idx_col])
        return df

    def to_huggingface_dataset(self):
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError("Install 'datasets' to use this export") from exc
        return Dataset.from_pandas(self.to_dataframe(include_labels=True))

    @property
    def stats(self) -> RunStats:
        if self._stats is not None:
            return self._stats
        status_counts: Dict[str, int] = {}
        if 'status' in self._df.columns:
            for code, count in self._df['status'].value_counts().items():
                try:
                    name = SampleStatus(int(code)).name
                except Exception:
                    name = f'UNKNOWN_{code}'
                status_counts[name] = int(count)
        n_tokens = int(self._sample_ptr[-1]) if self._sample_ptr is not None and len(self._sample_ptr) else 0
        hs_shape = None
        for path in ('hidden_states/per_token', 'response/hidden_states/data'):
            if path in self._zarr:
                hs_shape = tuple(self._zarr[path].shape)
                break
        n_ok = int(status_counts.get('OK', 0))
        n_error = int(status_counts.get('ERROR', 0) + status_counts.get('TIMEOUT', 0))
        self._stats = RunStats(
            n_samples_total=len(self),
            n_samples_ok=n_ok,
            n_samples_error=n_error,
            n_tokens_total=n_tokens,
            duration_seconds=self.manifest.stats.duration_seconds,
            run_id=self.manifest.run_id,
            status_counts=status_counts or None,
            model=self.manifest.model.name,
            dataset=self.manifest.dataset.name,
            n_layers=self.manifest.model.n_layers,
            hidden_dim=self.manifest.model.hidden_dim,
            hidden_states_shape=hs_shape,
            is_complete=self._is_complete,
        )
        return self._stats

    def validate(self) -> List[str]:
        issues: List[str] = []
        issues.extend(self.manifest.validate())
        for arr in ('tokens/ids', 'tokens/sample_ptr'):
            if arr not in self._zarr:
                issues.append(f'Missing required array: {arr}')
        if 'tokens/sample_ptr' in self._zarr:
            ptr = np.asarray(self._zarr['tokens/sample_ptr'][:])
            if len(ptr) > 1:
                if np.any(np.diff(ptr) < 0):
                    issues.append('Non-monotonic tokens/sample_ptr')
            if len(ptr) > 0 and 'tokens/ids' in self._zarr:
                n_tokens = int(self._zarr['tokens/ids'].shape[0])
                if int(ptr[-1]) != n_tokens:
                    issues.append(f'Token count mismatch: sample_ptr[-1]={int(ptr[-1])}, ids={n_tokens}')
            if 'tokens/sample_ptr' in self._zarr:
                n_samples_zarr = len(ptr) - 1
                n_samples_parquet = len(self._df)
                if n_samples_zarr != n_samples_parquet:
                    issues.append(f'Sample count mismatch: zarr={n_samples_zarr}, parquet={n_samples_parquet}')
        return issues

    def __repr__(self) -> str:
        state = 'complete' if self._is_complete else 'incomplete'
        return f"Run(path='{self.run_dir}', n_samples={len(self)}, n_valid={self.n_valid}, {state})"

    def __str__(self) -> str:
        mark = '✓' if self._is_complete else '⚠'
        return f'{mark} Run: {self.run_dir.name} ({self.n_valid}/{len(self)} valid)'


