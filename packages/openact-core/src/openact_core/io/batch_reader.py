from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TokenSelection:
    sample_idx: int
    flat_indices: np.ndarray
    local_indices: np.ndarray
    n_sample_tokens: int


class RunSnapshot:
    def __init__(self, run):
        self.run_dir = run.run_dir
        self._zarr = run._zarr
        self._df = run._df

        self.sample_ptr = np.asarray(self._zarr['tokens/sample_ptr'][:]).astype(np.int64)
        if 'sample_status' in self._zarr:
            self.sample_status = np.asarray(self._zarr['sample_status'][:]).astype(np.int64)
        else:
            self.sample_status = np.asarray(self._df.get('status', np.zeros(len(self._df), dtype=np.int64)))

        self.n_samples = int(len(self._df))
        self.hs_path = getattr(run, '_hs_per_token_path', None) or self._detect_hs_path()

        if self.hs_path and self.hs_path in self._zarr:
            hs_arr = self._zarr[self.hs_path]
            self.total_tokens = int(hs_arr.shape[0])
            self.n_layers = int(hs_arr.shape[1])
            self.hidden_dim = int(hs_arr.shape[2])
            self.hs_dtype = hs_arr.dtype
        else:
            self.total_tokens = int(self.sample_ptr[-1]) if len(self.sample_ptr) else 0
            if 'hidden_states/mean' in self._zarr:
                self.n_layers = int(self._zarr['hidden_states/mean'].shape[1])
                self.hidden_dim = int(self._zarr['hidden_states/mean'].shape[2])
            else:
                self.n_layers = 0
                self.hidden_dim = 0
            self.hs_dtype = None

    def _detect_hs_path(self) -> Optional[str]:
        for path in ('hidden_states/per_token', 'response/hidden_states/data'):
            if path in self._zarr:
                return path
        return None

    def get_token_range(self, sample_idx: int) -> Tuple[int, int]:
        i = int(sample_idx)
        if i + 1 >= len(self.sample_ptr):
            return (0, 0)
        return (int(self.sample_ptr[i]), int(self.sample_ptr[i + 1]))

    def get_n_tokens(self, sample_idx: int) -> int:
        s, e = self.get_token_range(sample_idx)
        return max(0, e - s)

    def get_valid_indices(self) -> np.ndarray:
        from openact_core.schema.status import SampleStatus

        return np.where(self.sample_status == int(SampleStatus.OK))[0].astype(np.int64)


class TokenSelector:
    @staticmethod
    def all_tokens(snap: RunSnapshot, sample_idx: int) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = max(0, e - s)
        local = np.arange(n, dtype=np.int64)
        return TokenSelection(int(sample_idx), local + s, local, n)

    @staticmethod
    def token_range(
        snap: RunSnapshot,
        sample_idx: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = max(0, e - s)
        lo = int(start) if start is not None else 0
        hi = int(end) if end is not None else n
        lo = max(0, lo)
        hi = min(n, hi)
        if lo >= hi:
            empty = np.array([], dtype=np.int64)
            return TokenSelection(int(sample_idx), empty, empty, n)
        local = np.arange(lo, hi, dtype=np.int64)
        return TokenSelection(int(sample_idx), local + s, local, n)

    @staticmethod
    def percentage_range(
        snap: RunSnapshot,
        sample_idx: int,
        start_pct: float = 0.0,
        end_pct: float = 1.0,
    ) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = max(0, e - s)
        lo = int(np.floor(n * float(start_pct)))
        hi = int(np.ceil(n * float(end_pct)))
        lo = max(0, lo)
        hi = min(n, hi)
        if lo >= hi:
            empty = np.array([], dtype=np.int64)
            return TokenSelection(int(sample_idx), empty, empty, n)
        local = np.arange(lo, hi, dtype=np.int64)
        return TokenSelection(int(sample_idx), local + s, local, n)

    @staticmethod
    def evenly_spaced(
        snap: RunSnapshot,
        sample_idx: int,
        n_points: int = 32,
    ) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = max(0, e - s)
        if n <= 0:
            empty = np.array([], dtype=np.int64)
            return TokenSelection(int(sample_idx), empty, empty, 0)
        if n_points <= 0:
            empty = np.array([], dtype=np.int64)
            return TokenSelection(int(sample_idx), empty, empty, n)
        grid = np.linspace(0, n - 1, num=min(n_points, max(1, n)))
        local = np.unique(np.clip(np.round(grid).astype(np.int64), 0, n - 1))
        return TokenSelection(int(sample_idx), local + s, local, n)


@dataclass
class ReadPlan:
    selections: List[TokenSelection]


def build_read_plan(selections: List[TokenSelection], merge_gap: int = 64) -> ReadPlan:
    del merge_gap
    return ReadPlan(selections=list(selections))


class BatchReader:
    def __init__(self, snap: RunSnapshot, layers: Optional[List[int]] = None, output_dtype: str = 'float32'):
        if snap.hs_path is None or snap.hs_path not in snap._zarr:
            raise ValueError('Per-token hidden states are not available in this run')
        self.snap = snap
        self.layers = layers
        self.output_dtype = np.dtype(output_dtype)
        self._hs_array = snap._zarr[snap.hs_path]

    def execute(self, plan: ReadPlan) -> Dict[int, np.ndarray]:
        results: Dict[int, np.ndarray] = {}
        layer_idx = self.layers
        for sel in plan.selections:
            if sel.n_sample_tokens <= 0 or len(sel.local_indices) == 0:
                L = len(layer_idx) if layer_idx is not None else self.snap.n_layers
                results[int(sel.sample_idx)] = np.zeros((0, L, self.snap.hidden_dim), dtype=self.output_dtype)
                continue
            s, e = self.snap.get_token_range(sel.sample_idx)
            block = np.asarray(self._hs_array[s:e])
            if block.size == 0:
                L = len(layer_idx) if layer_idx is not None else self.snap.n_layers
                results[int(sel.sample_idx)] = np.zeros((0, L, self.snap.hidden_dim), dtype=self.output_dtype)
                continue
            picked = block[sel.local_indices]
            if layer_idx is not None and picked.size:
                picked = picked[:, layer_idx, :]
            results[int(sel.sample_idx)] = picked.astype(self.output_dtype, copy=False)
        return results

    def execute_with_reduction(self, plan: ReadPlan, reduction: str = 'mean') -> Dict[int, np.ndarray]:
        raw = self.execute(plan)
        out: Dict[int, np.ndarray] = {}
        layer_idx = self.layers
        L = len(layer_idx) if layer_idx is not None else self.snap.n_layers
        zero = np.zeros((L, self.snap.hidden_dim), dtype=self.output_dtype)
        for idx, hs in raw.items():
            if hs.size == 0:
                out[int(idx)] = zero
                continue
            if reduction == 'mean':
                out[int(idx)] = hs.mean(axis=0)
            elif reduction == 'first':
                out[int(idx)] = hs[0]
            elif reduction == 'last':
                out[int(idx)] = hs[-1]
            else:
                out[int(idx)] = hs
        return out


def iter_batched_hidden_states(
    snap: RunSnapshot,
    sample_indices: List[int],
    selector_fn: Callable[[RunSnapshot, int], TokenSelection],
    layers: Optional[List[int]] = None,
    batch_size: int = 256,
    merge_gap: int = 64,
    output_dtype: str = 'float32',
):
    del merge_gap
    reader = BatchReader(snap, layers=layers, output_dtype=output_dtype)
    for start in range(0, len(sample_indices), max(1, int(batch_size))):
        batch = sample_indices[start : start + max(1, int(batch_size))]
        selections = [selector_fn(snap, int(i)) for i in batch]
        plan = build_read_plan(selections)
        data = reader.execute(plan)
        for sel in selections:
            hs = data.get(int(sel.sample_idx))
            if hs is not None:
                yield int(sel.sample_idx), sel, hs


class HiddenStateLoader:
    def __init__(
        self,
        run,
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
        output_dtype: str = 'float32',
    ):
        self.snap = RunSnapshot(run)
        self.layers = layers
        self.only_valid = bool(only_valid)
        self.output_dtype = output_dtype
        self._indices = self.snap.get_valid_indices() if only_valid else np.arange(self.snap.n_samples, dtype=np.int64)

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def n_effective_layers(self) -> int:
        return len(self.layers) if self.layers is not None else int(self.snap.n_layers)

    def load_full_sequences(self, batch_size: int = 256) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for idx, _sel, hs in iter_batched_hidden_states(
            self.snap,
            self._indices.tolist(),
            selector_fn=TokenSelector.all_tokens,
            layers=self.layers,
            batch_size=batch_size,
            output_dtype=self.output_dtype,
        ):
            result[int(idx)] = hs
        return result

    def load_range_mean(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        batch_size: int = 512,
    ) -> np.ndarray:
        if start_pct is not None or end_pct is not None:
            sp = float(start_pct) if start_pct is not None else 0.0
            ep = float(end_pct) if end_pct is not None else 1.0
            selector = lambda snap, idx: TokenSelector.percentage_range(snap, idx, sp, ep)
        else:
            selector = lambda snap, idx: TokenSelector.token_range(snap, idx, start, end)

        reader = BatchReader(self.snap, layers=self.layers, output_dtype=self.output_dtype)
        out: List[np.ndarray] = []
        layer_idx = self.layers
        L = len(layer_idx) if layer_idx is not None else self.snap.n_layers
        zero = np.zeros((L, self.snap.hidden_dim), dtype=self.output_dtype)

        for start_i in range(0, len(self._indices), max(1, int(batch_size))):
            batch = self._indices[start_i : start_i + max(1, int(batch_size))].tolist()
            sels = [selector(self.snap, int(i)) for i in batch]
            reduced = reader.execute_with_reduction(ReadPlan(sels), reduction='mean')
            for i in batch:
                out.append(reduced.get(int(i), zero))
        if not out:
            return np.empty((0, L, self.snap.hidden_dim), dtype=self.output_dtype)
        return np.stack(out, axis=0)

    def load_aligned_trajectories(
        self,
        n_points: int = 32,
        batch_size: int = 256,
    ) -> np.ndarray:
        selector = lambda snap, idx: TokenSelector.evenly_spaced(snap, idx, n_points=n_points)
        reader = BatchReader(self.snap, layers=self.layers, output_dtype=self.output_dtype)
        out: List[np.ndarray] = []
        layer_idx = self.layers
        L = len(layer_idx) if layer_idx is not None else self.snap.n_layers
        zero = np.zeros((n_points, L, self.snap.hidden_dim), dtype=self.output_dtype)

        for start_i in range(0, len(self._indices), max(1, int(batch_size))):
            batch = self._indices[start_i : start_i + max(1, int(batch_size))].tolist()
            sels = [selector(self.snap, int(i)) for i in batch]
            data = reader.execute(ReadPlan(sels))
            for sel in sels:
                hs = data.get(int(sel.sample_idx))
                if hs is None or hs.size == 0:
                    out.append(zero)
                    continue
                if hs.shape[0] < n_points:
                    padded = np.zeros((n_points, hs.shape[1], hs.shape[2]), dtype=hs.dtype)
                    padded[: hs.shape[0]] = hs
                    out.append(padded)
                else:
                    out.append(hs[:n_points])
        if not out:
            return np.empty((0, n_points, L, self.snap.hidden_dim), dtype=self.output_dtype)
        return np.stack(out, axis=0)
