from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import zarr


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
        self.sample_ptr = np.asarray(run._zarr['tokens/sample_ptr'][:])
        self.sample_status = np.asarray(run._zarr['sample_status'][:])
        self.n_samples = len(self._df)
        self.hs_path = self._detect_hs_path()
        if self.hs_path:
            hs_arr = self._zarr[self.hs_path]
            self.total_tokens = hs_arr.shape[0]
            self.n_layers = hs_arr.shape[1]
            self.hidden_dim = hs_arr.shape[2]
            self.hs_dtype = hs_arr.dtype
        else:
            self.total_tokens = int(self.sample_ptr[-1])
            if 'hidden_states/mean' in self._zarr:
                self.n_layers = self._zarr['hidden_states/mean'].shape[1]
                self.hidden_dim = self._zarr['hidden_states/mean'].shape[2]
            else:
                self.n_layers = 0
                self.hidden_dim = 0
            self.hs_dtype = None
    
    def _detect_hs_path(self) -> Optional[str]:
        for path in ['hidden_states/per_token', 'response/hidden_states/data']:
            if path in self._zarr:
                return path
        return None
    
    def get_token_range(self, sample_idx: int) -> Tuple[int, int]:
        return (int(self.sample_ptr[sample_idx]), int(self.sample_ptr[sample_idx + 1]))
    
    def get_n_tokens(self, sample_idx: int) -> int:
        s, e = self.get_token_range(sample_idx)
        return e - s
    
    def get_valid_indices(self) -> np.ndarray:
        from openact_core.schema.status import SampleStatus
        return np.where(self.sample_status == SampleStatus.OK)[0]


class TokenSelector:
    @staticmethod
    def all_tokens(snap: RunSnapshot, sample_idx: int) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = e - s
        return TokenSelection(
            sample_idx=sample_idx,
            flat_indices=np.arange(s, e),
            local_indices=np.arange(n),
            n_sample_tokens=n,
        )
    
    @staticmethod
    def token_range(snap: RunSnapshot, sample_idx: int, start: Optional[int] = None, end: Optional[int] = None) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = e - s
        lo = start or 0
        hi = min(end, n) if end is not None else n
        lo = max(0, lo)
        if lo >= hi:
            return TokenSelection(sample_idx, np.array([], dtype=int), np.array([], dtype=int), n)
        local = np.arange(lo, hi)
        return TokenSelection(sample_idx, local + s, local, n)
    
    @staticmethod
    def percentage_range(snap: RunSnapshot, sample_idx: int, start_pct: float = 0.0, end_pct: float = 1.0) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = e - s
        lo = int(n * start_pct)
        hi = int(n * end_pct)
        lo, hi = max(0, lo), min(n, hi)
        if lo >= hi:
            return TokenSelection(sample_idx, np.array([], dtype=int), np.array([], dtype=int), n)
        local = np.arange(lo, hi)
        return TokenSelection(sample_idx, local + s, local, n)
    
    @staticmethod
    def evenly_spaced(snap: RunSnapshot, sample_idx: int, n_points: int = 32) -> TokenSelection:
        s, e = snap.get_token_range(sample_idx)
        n = e - s
        if n <= n_points:
            return TokenSelector.all_tokens(snap, sample_idx)
        local = np.linspace(0, n - 1, n_points, dtype=int)
        local = np.unique(local)
        return TokenSelection(sample_idx, local + s, local, n)


@dataclass
class ReadPlan:
    intervals: List[Tuple[int, int]]
    selections: List[TokenSelection]
    total_tokens_to_read: int
    n_intervals: int


def build_read_plan(selections: List[TokenSelection], merge_gap: int = 64) -> ReadPlan:
    all_positions = set()
    for sel in selections:
        all_positions.update(sel.flat_indices.tolist())
    
    if not all_positions:
        return ReadPlan([], selections, 0, 0)
    
    sorted_pos = sorted(all_positions)
    intervals = []
    cur_start = sorted_pos[0]
    cur_end = sorted_pos[0] + 1
    
    for pos in sorted_pos[1:]:
        if pos <= cur_end + merge_gap:
            cur_end = pos + 1
        else:
            intervals.append((cur_start, cur_end))
            cur_start = pos
            cur_end = pos + 1
    intervals.append((cur_start, cur_end))
    
    return ReadPlan(intervals, selections, len(sorted_pos), len(intervals))


class BatchReader:
    def __init__(self, snap: RunSnapshot, layers: Optional[List[int]] = None, output_dtype: str = 'float32'):
        self.snap = snap
        self.layers = layers
        self.output_dtype = np.dtype(output_dtype)
        if snap.hs_path is None:
            raise ValueError("No per-token hidden states in this run")
        self._hs_array = snap._zarr[snap.hs_path]
    
    def execute(self, plan: ReadPlan) -> Dict[int, np.ndarray]:
        if not plan.intervals:
            return {}
        
        cache = {}
        for iv_start, iv_end in plan.intervals:
            block = np.asarray(self._hs_array[iv_start:iv_end])
            if self.layers is not None:
                block = block[:, self.layers, :]
            if block.dtype != self.output_dtype:
                block = block.astype(self.output_dtype)
            cache[(iv_start, iv_end)] = block
        
        results = {}
        for sel in plan.selections:
            if len(sel.flat_indices) == 0:
                L = len(self.layers) if self.layers else self.snap.n_layers
                results[sel.sample_idx] = np.zeros((0, L, self.snap.hidden_dim), dtype=self.output_dtype)
                continue
            
            token_vectors = []
            for flat_idx in sel.flat_indices:
                for (iv_start, iv_end), block in cache.items():
                    if iv_start <= flat_idx < iv_end:
                        token_vectors.append(block[flat_idx - iv_start])
                        break
            
            results[sel.sample_idx] = np.stack(token_vectors, axis=0)
        
        return results
    
    def execute_with_reduction(self, plan: ReadPlan, reduction: str = 'mean') -> Dict[int, np.ndarray]:
        raw = self.execute(plan)
        reduced = {}
        for sample_idx, hs in raw.items():
            if len(hs) == 0:
                L = len(self.layers) if self.layers else self.snap.n_layers
                reduced[sample_idx] = np.zeros((L, self.snap.hidden_dim), dtype=self.output_dtype)
                continue
            
            if reduction == 'mean':
                reduced[sample_idx] = hs.mean(axis=0)
            elif reduction == 'first':
                reduced[sample_idx] = hs[0]
            elif reduction == 'last':
                reduced[sample_idx] = hs[-1]
            else:
                reduced[sample_idx] = hs
        
        return reduced


def iter_batched_hidden_states(
    snap: RunSnapshot,
    sample_indices: List[int],
    selector_fn: Callable[[RunSnapshot, int], TokenSelection],
    layers: Optional[List[int]] = None,
    batch_size: int = 256,
    merge_gap: int = 64,
    output_dtype: str = 'float32',
):
    reader = BatchReader(snap, layers=layers, output_dtype=output_dtype)
    for batch_start in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[batch_start:batch_start + batch_size]
        selections = [selector_fn(snap, idx) for idx in batch_indices]
        plan = build_read_plan(selections, merge_gap=merge_gap)
        results = reader.execute(plan)
        for sel in selections:
            hs = results.get(sel.sample_idx)
            if hs is not None:
                yield sel.sample_idx, sel, hs


class HiddenStateLoader:
    def __init__(self, run, layers: Optional[List[int]] = None, only_valid: bool = True, output_dtype: str = 'float32'):
        self.snap = RunSnapshot(run)
        self.layers = layers
        self.only_valid = only_valid
        self.output_dtype = output_dtype
        self._indices = self.snap.get_valid_indices() if only_valid else np.arange(self.snap.n_samples)
    
    @property
    def n_effective_layers(self) -> int:
        return len(self.layers) if self.layers else self.snap.n_layers
    
    def load_trajectories(self, batch_size: int = 256) -> Dict[int, np.ndarray]:
        result = {}
        for idx, sel, hs in iter_batched_hidden_states(
            self.snap, self._indices.tolist(),
            selector_fn=TokenSelector.all_tokens,
            layers=self.layers,
            batch_size=batch_size,
            output_dtype=self.output_dtype,
        ):
            result[idx] = hs
        return result
    
    def load_range_mean(self, start: Optional[int] = None, end: Optional[int] = None, start_pct: Optional[float] = None, end_pct: Optional[float] = None, batch_size: int = 512) -> np.ndarray:
        use_pct = start_pct is not None or end_pct is not None
        if use_pct:
            sp = start_pct or 0.0
            ep = end_pct or 1.0
            selector = lambda snap, idx: TokenSelector.percentage_range(snap, idx, sp, ep)
        else:
            selector = lambda snap, idx: TokenSelector.token_range(snap, idx, start, end)
        
        results = {}
        for idx, sel, hs in iter_batched_hidden_states(
            self.snap, self._indices.tolist(),
            selector_fn=selector,
            layers=self.layers,
            batch_size=batch_size,
            output_dtype=self.output_dtype,
        ):
            results[idx] = hs.mean(axis=0) if len(hs) > 0 else np.zeros((self.n_effective_layers, self.snap.hidden_dim), dtype=self.output_dtype)
        
        return np.stack([results[int(idx)] for idx in self._indices], axis=0)
    
    def load_aligned_trajectories(self, n_points: int = 32, batch_size: int = 256) -> np.ndarray:
        selector = lambda snap, idx: TokenSelector.evenly_spaced(snap, idx, n_points)
        results = {}
        for idx, sel, hs in iter_batched_hidden_states(
            self.snap, self._indices.tolist(),
            selector_fn=selector,
            layers=self.layers,
            batch_size=batch_size,
            output_dtype=self.output_dtype,
        ):
            if len(hs) < n_points:
                padded = np.zeros((n_points, self.n_effective_layers, self.snap.hidden_dim), dtype=self.output_dtype)
                padded[:len(hs)] = hs
                results[idx] = padded
            else:
                results[idx] = hs[:n_points]
        
        return np.stack([results[int(idx)] for idx in self._indices], axis=0)
