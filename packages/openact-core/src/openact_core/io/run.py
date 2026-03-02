"""
Run reader — the primary entry point for loading an OpenAct run.

A :class:`Run` represents a completed (or in-progress) data-collection run.
It provides indexed access to individual :class:`Sample` objects and bulk
accessors for hidden states, labels, and metadata.
"""

from pathlib import Path
from typing import Union, Optional, List, Iterator, Dict, Any, Callable
import warnings
import numpy as np
import pandas as pd
import zarr

from openact_core.schema.manifest import Manifest
from openact_core.schema.status import SampleStatus
from openact_core.io.sample import Sample
from openact_core.io.run_stats import RunStats


def _to_python_native(value: Any) -> Any:
    """Convert numpy scalars to plain Python types for JSON compatibility."""
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.str_,)):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Label index — lazy reader for sidecar parquet label files
# ---------------------------------------------------------------------------

class _LabelIndex:

    def __init__(self, labels_dir: Path):
        self._labels_dir = labels_dir
        self._frames: Dict[str, pd.DataFrame] = {}
        self._col_to_file: Dict[str, str] = {}
        self._indexed: Dict[str, Dict[int, int]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._labels_dir.exists():
            return
        for label_file in self._labels_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(label_file)
                self._frames[label_file.stem] = df
                idx_col = (
                    "sample_idx" if "sample_idx" in df.columns else "index"
                )
                if idx_col in df.columns:
                    row_map = {}
                    for row_idx, sample_idx_val in enumerate(df[idx_col]):
                        row_map[int(sample_idx_val)] = row_idx
                    self._indexed[label_file.stem] = row_map
                for col in df.columns:
                    if col not in ("sample_idx", "index"):
                        self._col_to_file[col] = label_file.stem
            except Exception:
                continue

    def get(self, sample_idx: int, label_name: str) -> Any:
        self._ensure_loaded()
        file_stem = self._col_to_file.get(label_name)
        if file_stem is None:
            return None
        df = self._frames.get(file_stem)
        if df is None:
            return None
        idx_map = self._indexed.get(file_stem)
        if idx_map is None:
            return None
        row_idx = idx_map.get(sample_idx)
        if row_idx is None:
            return None
        value = df.iloc[row_idx][label_name]
        if pd.isna(value):
            return None
        return _to_python_native(value)

    def get_column(self, label_name: str) -> Optional[pd.Series]:
        self._ensure_loaded()
        file_stem = self._col_to_file.get(label_name)
        if file_stem is None:
            return None
        df = self._frames.get(file_stem)
        if df is None:
            return None
        if label_name not in df.columns:
            return None
        return df[label_name]

    def get_column_indexed(self, label_name: str) -> Optional[Dict[int, Any]]:
        self._ensure_loaded()
        file_stem = self._col_to_file.get(label_name)
        if file_stem is None:
            return None
        df = self._frames.get(file_stem)
        if df is None:
            return None
        idx_map = self._indexed.get(file_stem)
        if idx_map is None:
            return None
        result = {}
        for sample_idx, row_idx in idx_map.items():
            val = df.iloc[row_idx][label_name]
            if pd.isna(val):
                result[sample_idx] = None
            else:
                result[sample_idx] = _to_python_native(val)
        return result

    def list_columns(self) -> List[str]:
        self._ensure_loaded()
        return list(self._col_to_file.keys())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class Run:
    """Reader for a single OpenAct data-collection run.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run directory (must contain ``tensors.zarr`` and either
        ``data.parquet`` or a legacy ``parquet/records`` directory).
    require_complete : bool
        If ``True``, raise when the ``_SUCCESS`` marker is absent.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        require_complete: bool = False,
    ):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        # Completion marker
        success_file = self.run_dir / "_SUCCESS"
        self._is_complete = success_file.exists()
        self._completion_info: Optional[Dict[str, Any]] = None
        if self._is_complete:
            try:
                import json

                with open(success_file) as f:
                    self._completion_info = json.load(f)
            except Exception:
                pass

        if require_complete and not self._is_complete:
            raise ValueError(
                f"Run at {run_dir} is not marked as complete. It may be "
                f"incomplete or still being written. Use "
                f"require_complete=False to load anyway."
            )
        if not self._is_complete:
            warnings.warn(
                f"Run at {run_dir} has no _SUCCESS marker. "
                f"Data may be incomplete or corrupted.",
                UserWarning,
            )

        # Manifest
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            self.manifest = Manifest.load(manifest_path)
        else:
            self.manifest = Manifest()

        # Zarr tensor store
        zarr_path = self.run_dir / "tensors.zarr"
        if not zarr_path.exists():
            raise ValueError(f"Zarr storage not found: {zarr_path}")
        self._zarr = zarr.open_group(str(zarr_path), mode="r")

        # Metadata parquet
        parquet_path = self.run_dir / "data.parquet"
        if parquet_path.exists():
            self._df = pd.read_parquet(parquet_path)
        else:
            legacy_path = self.run_dir / "parquet" / "records"
            if legacy_path.exists():
                import pyarrow.dataset as ds

                dataset = ds.dataset(legacy_path, format="parquet")
                self._df = dataset.to_table().to_pandas()
            else:
                raise ValueError(
                    f"No metadata parquet found in {self.run_dir}"
                )

        # Sidecar label files
        self._label_index = _LabelIndex(self.run_dir / "labels")

        # Cached stats (built lazily on first access)
        self._stats_cache: Optional[RunStats] = None

    # -- Basic properties ----------------------------------------------------

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def completion_info(self) -> Optional[Dict[str, Any]]:
        return self._completion_info

    # -- Container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return Sample(self, idx)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, idx: int) -> bool:
        return 0 <= idx < len(self)

    # -- Iteration helpers ---------------------------------------------------

    def iter_valid(self) -> Iterator[Sample]:
        """Yield only samples with status OK."""
        for i in range(len(self)):
            if self._df.iloc[i]["status"] == SampleStatus.OK:
                yield self[i]

    def iter_by_status(self, status: SampleStatus) -> Iterator[Sample]:
        """Yield samples matching *status*."""
        for i in range(len(self)):
            if self._df.iloc[i]["status"] == status:
                yield self[i]

    def filter(self, predicate: Callable[[Sample], bool]) -> List[Sample]:
        """Return valid samples for which *predicate* returns ``True``."""
        return [s for s in self.iter_valid() if predicate(s)]

    def filter_by_label(self, label_name: str, value: Any) -> List[Sample]:
        """Return valid samples where ``get_label(label_name) == value``."""
        return self.filter(lambda s: s.get_label(label_name) == value)

    # -- Index helpers -------------------------------------------------------

    def get_indices_by_status(self, status: SampleStatus) -> np.ndarray:
        mask = self._df["status"] == status
        return np.where(mask)[0]

    def get_valid_indices(self) -> np.ndarray:
        return self.get_indices_by_status(SampleStatus.OK)

    # -- Bulk hidden-state access --------------------------------------------

    def get_all_hidden_states(
        self,
        reduction: str = "mean",
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
    ) -> np.ndarray:
        """Load hidden states for all (valid) samples with *reduction*.

        Uses the batch reader when per-token states are available, falling
        back to per-sample iteration otherwise.
        """
        try:
            from openact_core.io.batch_reader import (
                HiddenStateLoader,
                TokenSelector,
                iter_batched_hidden_states,
            )

            loader = HiddenStateLoader(
                self, layers=layers, only_valid=only_valid
            )
            if reduction == "mean":
                return loader.load_range_mean()
            elif reduction == "first":
                selector = lambda snap, idx: TokenSelector.token_range(
                    snap, idx, start=0, end=1
                )
            elif reduction == "last":
                selector = lambda snap, idx: TokenSelector.token_range(
                    snap,
                    idx,
                    start=snap.get_n_tokens(idx) - 1,
                    end=snap.get_n_tokens(idx),
                )
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

            if reduction in ("first", "last"):
                results = {}
                for idx, sel, hs in iter_batched_hidden_states(
                    loader.snap,
                    loader._indices.tolist(),
                    selector_fn=selector,
                    layers=layers,
                    batch_size=512,
                ):
                    results[idx] = (
                        hs[0]
                        if len(hs) > 0
                        else np.zeros(
                            (loader.n_effective_layers, loader.snap.hidden_dim),
                            dtype="float32",
                        )
                    )
                return np.stack(
                    [results[int(idx)] for idx in loader._indices], axis=0
                )

        except (ImportError, ValueError, AttributeError):
            # Fallback: iterate sample-by-sample
            results = []
            iterator = self.iter_valid() if only_valid else self
            for sample in iterator:
                hs = sample.hidden_states
                if len(hs) == 0:
                    continue
                if reduction == "mean":
                    hs = hs.mean(axis=0)
                elif reduction == "first":
                    hs = hs[0]
                elif reduction == "last":
                    hs = hs[-1]
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")
                if layers is not None:
                    hs = hs[layers, :]
                results.append(hs)
            if not results:
                return np.array([])
            return np.stack(results, axis=0)

    def get_all_mean_hidden_states(
        self,
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
    ) -> np.ndarray:
        """Shortcut for pre-computed per-sample mean hidden states."""
        if "hidden_states/mean" in self._zarr:
            data = np.asarray(self._zarr["hidden_states/mean"][:])
            if only_valid:
                valid_mask = self._df["status"] == SampleStatus.OK
                data = data[valid_mask.values]
            if layers is not None:
                data = data[:, layers, :]
            return data
        return self.get_all_hidden_states(
            reduction="mean", layers=layers, only_valid=only_valid
        )

    def get_all_prompt_last_hidden_states(
        self,
        layers: Optional[List[int]] = None,
        only_valid: bool = True,
    ) -> Optional[np.ndarray]:
        """Return the last-prompt-token hidden states, if available."""
        if "hidden_states/prompt_last" not in self._zarr:
            return None
        data = np.asarray(self._zarr["hidden_states/prompt_last"][:])
        if only_valid:
            valid_mask = self._df["status"] == SampleStatus.OK
            data = data[valid_mask.values]
        if layers is not None:
            data = data[:, layers, :]
        return data

    def get_all_generation_metrics(
        self, only_valid: bool = True
    ) -> pd.DataFrame:
        cols = ["sample_idx", "perplexity", "entropy", "max_probability"]
        available_cols = [c for c in cols if c in self._df.columns]
        df = self._df[available_cols].copy()
        if only_valid:
            df = df[self._df["status"] == SampleStatus.OK]
        return df

    def get_all_token_ids(self, only_valid: bool = True) -> List[np.ndarray]:
        results = []
        iterator = self.iter_valid() if only_valid else self
        for sample in iterator:
            results.append(sample.token_ids)
        return results

    # -- Label access --------------------------------------------------------

    def get_label(self, sample_idx: int, label_name: str) -> Any:
        if label_name in self._df.columns:
            value = self._df.iloc[sample_idx][label_name]
            if pd.isna(value):
                return None
            return _to_python_native(value)
        return self._label_index.get(sample_idx, label_name)

    def get_all_labels(
        self, label_name: str, only_valid: bool = True
    ) -> np.ndarray:
        indices = (
            self.get_valid_indices() if only_valid else np.arange(len(self))
        )

        # Fast path: column lives in the main parquet
        if label_name in self._df.columns:
            values = self._df[label_name].values
            selected = values[indices]
            result = []
            for v in selected:
                if pd.isna(v):
                    result.append(None)
                else:
                    result.append(_to_python_native(v))
            return np.array(result, dtype=object)

        # Try sidecar parquet
        indexed_col = self._label_index.get_column_indexed(label_name)
        if indexed_col is not None:
            result = [indexed_col.get(int(i)) for i in indices]
            return np.array(result, dtype=object)

        # Fallback: per-sample lookup
        labels = [self.get_label(int(i), label_name) for i in indices]
        return np.array(labels, dtype=object)

    def list_available_labels(self) -> List[str]:
        labels = set(self._df.columns)
        labels.update(self._label_index.list_columns())
        internal = {"sample_idx", "index", "sample_id"}
        return sorted(labels - internal)

    # -- Stats (cached) ------------------------------------------------------

    @property
    def stats(self) -> RunStats:
        """Aggregate statistics for this run.

        The result is computed once and cached for subsequent accesses so that
        repeated calls (e.g. in dashboards) do not re-scan the zarr store.
        """
        if self._stats_cache is not None:
            return self._stats_cache

        status_counts = self._df["status"].value_counts().to_dict()
        status_names: Dict[str, int] = {}
        for code, count in status_counts.items():
            try:
                name = SampleStatus(int(code)).name
            except ValueError:
                name = f"UNKNOWN_{code}"
            status_names[name] = int(count)

        n_tokens = 0
        if "tokens/sample_ptr" in self._zarr:
            ptr = self._zarr["tokens/sample_ptr"]
            n_tokens = int(ptr[-1]) if len(ptr) > 0 else 0

        hs_shape = None
        if "hidden_states/per_token" in self._zarr:
            hs_shape = self._zarr["hidden_states/per_token"].shape
        elif "response/hidden_states/data" in self._zarr:
            hs_shape = self._zarr["response/hidden_states/data"].shape

        n_ok = status_names.get("OK", 0)
        n_error = status_names.get("ERROR", 0) + status_names.get("TIMEOUT", 0)
        duration = self.manifest.stats.duration_seconds

        self._stats_cache = RunStats(
            n_samples_total=len(self),
            n_samples_ok=n_ok,
            n_samples_error=n_error,
            n_tokens_total=n_tokens,
            duration_seconds=duration,
            run_id=self.manifest.run_id,
            status_counts=status_names,
            model=self.manifest.model.name,
            dataset=self.manifest.dataset.name,
            n_layers=self.manifest.model.n_layers,
            hidden_dim=self.manifest.model.hidden_dim,
            hidden_states_shape=hs_shape,
            is_complete=self._is_complete,
        )
        return self._stats_cache

    # -- Convenience properties ----------------------------------------------

    @property
    def n_total(self) -> int:
        """Total number of samples in the run (same as len(self))."""
        return len(self)

    @property
    def n_valid(self) -> int:
        return int((self._df["status"] == SampleStatus.OK).sum())

    @property
    def n_layers(self) -> Optional[int]:
        if "hidden_states/per_token" in self._zarr:
            return self._zarr["hidden_states/per_token"].shape[1]
        if "hidden_states/mean" in self._zarr:
            return self._zarr["hidden_states/mean"].shape[1]
        if "response/hidden_states/data" in self._zarr:
            return self._zarr["response/hidden_states/data"].shape[1]
        return self.manifest.model.n_layers

    @property
    def hidden_dim(self) -> Optional[int]:
        if "hidden_states/per_token" in self._zarr:
            return self._zarr["hidden_states/per_token"].shape[2]
        if "hidden_states/mean" in self._zarr:
            return self._zarr["hidden_states/mean"].shape[2]
        if "response/hidden_states/data" in self._zarr:
            return self._zarr["response/hidden_states/data"].shape[2]
        return self.manifest.model.hidden_dim

    # -- Export --------------------------------------------------------------

    def to_dataframe(self, include_labels: bool = True) -> pd.DataFrame:
        """Return all sample metadata as a DataFrame.

        When *include_labels* is ``True``, sidecar label parquets are merged
        in via a left join on ``sample_idx``.
        """
        df = self._df.copy()
        if include_labels:
            self._label_index._ensure_loaded()
            for file_stem, label_df in self._label_index._frames.items():
                try:
                    idx_col = (
                        "sample_idx"
                        if "sample_idx" in label_df.columns
                        else "index"
                    )
                    if idx_col in label_df.columns and idx_col in df.columns:
                        df = df.merge(
                            label_df,
                            on=idx_col,
                            how="left",
                            suffixes=("", f"_{file_stem}"),
                        )
                except Exception:
                    pass
        return df

    def to_huggingface_dataset(self):
        """Convert the run metadata to a HuggingFace ``Dataset``."""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Install 'datasets' package: pip install datasets"
            )
        df = self.to_dataframe()
        return Dataset.from_pandas(df)

    # -- Validation ----------------------------------------------------------

    def validate(self) -> List[str]:
        """Run lightweight integrity checks; return a list of issues."""
        issues = []
        issues.extend(self.manifest.validate())

        required_arrays = ["tokens/ids", "tokens/sample_ptr"]
        for arr in required_arrays:
            if arr not in self._zarr:
                issues.append(f"Missing required array: {arr}")

        if "tokens/sample_ptr" in self._zarr:
            ptr = self._zarr["tokens/sample_ptr"][:]
            for i in range(len(ptr) - 1):
                if ptr[i] > ptr[i + 1]:
                    issues.append(
                        f"Non-monotonic pointer at {i}: "
                        f"{ptr[i]} > {ptr[i + 1]}"
                    )

        if "tokens/sample_ptr" in self._zarr:
            n_samples_zarr = len(self._zarr["tokens/sample_ptr"]) - 1
            n_samples_parquet = len(self._df)
            if n_samples_zarr != n_samples_parquet:
                issues.append(
                    f"Sample count mismatch: "
                    f"zarr={n_samples_zarr}, parquet={n_samples_parquet}"
                )
        return issues

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        complete_str = "complete" if self._is_complete else "incomplete"
        return (
            f"Run(path='{self.run_dir}', n_samples={len(self)}, "
            f"n_valid={self.n_valid}, {complete_str})"
        )

    def __str__(self) -> str:
        status = "✓" if self._is_complete else "⚠"
        return (
            f"{status} Run: {self.run_dir.name} "
            f"({self.n_valid}/{len(self)} valid)"
        )