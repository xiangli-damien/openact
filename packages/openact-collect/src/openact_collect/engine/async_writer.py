"""
AsyncZarrWriter — threaded writer for streaming hidden-state data to Zarr.

Improvements over the original:
  - Proactive worker health checking: callers learn about worker failures
    immediately (on the next submit) instead of only at finalize().
  - Queue draining on worker error to prevent deadlocks.
  - Context-manager interface (__enter__ / __exit__).
  - Retry-with-health-check loop in submit() to avoid blocking forever on
    a dead worker.
"""

import queue
import threading
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import zarr
from numcodecs import Blosc

from openact_collect.schema import CaptureSpec, StorageSpec
from openact_core.schema.manifest import Manifest
from openact_core.schema.status import SampleStatus


class WriteTaskType(Enum):
    SAMPLE = "sample"
    FLUSH = "flush"
    FINALIZE = "finalize"


@dataclass
class WriteTask:
    task_type: WriteTaskType
    data: Optional[Dict[str, Any]] = None


class AsyncZarrWriter:
    """Asynchronous writer that streams sample data into a Zarr directory store.

    The writer owns a single background thread that consumes ``WriteTask``
    objects from an internal queue.  This keeps the main collection loop
    free from I/O blocking.

    Usage::

        writer = AsyncZarrWriter(zarr_path, manifest, capture_spec, storage_spec)
        writer.start()
        writer.submit_sample(...)
        writer.finalize()

    Or as a context manager::

        with AsyncZarrWriter(zarr_path, manifest, capture_spec, storage_spec) as w:
            w.submit_sample(...)
        # finalize() called automatically
    """

    def __init__(
        self,
        zarr_path: Path,
        manifest: Manifest,
        capture_spec: CaptureSpec,
        storage_spec: StorageSpec,
        queue_size: int = 8,
    ):
        self.zarr_path = Path(zarr_path)
        self.manifest = manifest
        self.capture_spec = capture_spec
        self.storage_spec = storage_spec
        self._max_queue_size = queue_size

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._root: Optional[zarr.Group] = None
        self._arrays: Dict[str, zarr.Array] = {}
        self._current_token_ptr = 0
        self._initialized = False
        self._lock = threading.Lock()
        self._allocated_samples = 0

        # Error propagation
        self._worker_error: Optional[BaseException] = None
        self._worker_error_lock = threading.Lock()

    # -- Context-manager interface -----------------------------------------

    def __enter__(self) -> "AsyncZarrWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None:
                self.finalize()
            else:
                # On exception: still try to flush what we can, but don't
                # raise a secondary error that would mask the original.
                if self._initialized:
                    self._stop_event.set()
                    if self._worker is not None and self._worker.is_alive():
                        self._drain_queue()
                        self._worker.join(timeout=10)
        except Exception as cleanup_err:
            import logging

            logging.getLogger("openact.collect").error(
                "Error during AsyncZarrWriter cleanup: %s", cleanup_err
            )
        return False

    # -- Initialisation ----------------------------------------------------

    def start(self) -> None:
        if self._initialized:
            return
        self._initialize_zarr()
        self._worker = threading.Thread(
            target=self._write_loop, daemon=True, name="ZarrWriter"
        )
        self._worker.start()
        self._initialized = True

    def _initialize_zarr(self) -> None:
        # mode='w' overwrites existing store so reruns are idempotent; no need to
        # manually clear the directory or risk "shape do not match" from stale zarr.
        self.zarr_path.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(
            cname=self.storage_spec.compression,
            clevel=self.storage_spec.compression_level,
        )
        self._root = zarr.open_group(str(self.zarr_path), mode="w")

        n_samples = self.manifest.dataset.n_samples or 10000
        self._allocated_samples = n_samples

        chunk_t = int(self.storage_spec.chunk_tokens)
        chunk_l = int(getattr(self.storage_spec, "chunk_layers", 0) or 0)
        chunk_h = int(getattr(self.storage_spec, "chunk_hidden_dim", 0) or 0)
        n_layers = self.manifest.model.n_layers or 32
        hidden_dim = self.manifest.model.hidden_dim or 4096

        if self.capture_spec.hidden_states_layers is not None:
            effective_layers = len(self.capture_spec.hidden_states_layers)
        else:
            effective_layers = n_layers

        # -- Token-level arrays --------------------------------------------
        tokens_grp = self._root.require_group("tokens")

        self._arrays["token_ids"] = tokens_grp.require_dataset(
            "ids",
            shape=(0,),
            chunks=(chunk_t * 16,),
            dtype="int32",
            compressor=compressor,
        )
        self._arrays["token_offsets"] = tokens_grp.require_dataset(
            "offsets",
            shape=(0, 2),
            chunks=(chunk_t * 16, 2),
            dtype="int32",
            compressor=compressor,
        )

        sample_chunk = min(4096, n_samples + 1)
        self._arrays["sample_ptr"] = tokens_grp.require_dataset(
            "sample_ptr",
            shape=(n_samples + 1,),
            chunks=(sample_chunk,),
            dtype="int64",
            fill_value=-1,
        )
        self._arrays["sample_ptr"][0] = 0

        status_chunk = min(4096, n_samples)
        self._arrays["sample_status"] = self._root.require_dataset(
            "sample_status",
            shape=(n_samples,),
            chunks=(status_chunk,),
            dtype="int8",
            fill_value=int(SampleStatus.UNPROCESSED),
        )

        # -- Hidden-state arrays -------------------------------------------
        if self.capture_spec.hidden_states:
            hs_grp = self._root.require_group("hidden_states")
            layers = self.capture_spec.hidden_states_layers or list(range(n_layers))
            hs_grp.attrs["layers"] = layers
            hs_grp.attrs["dtype"] = self.capture_spec.hidden_states_dtype

            hs_dtype = self.capture_spec.hidden_states_dtype

            if self.capture_spec.save_per_token:
                eff_chunk_l = max(
                    1, min(chunk_l if chunk_l > 0 else effective_layers, effective_layers)
                )
                eff_chunk_h = max(
                    1, min(chunk_h if chunk_h > 0 else hidden_dim, hidden_dim)
                )
                self._arrays["hs_per_token"] = hs_grp.require_dataset(
                    "per_token",
                    shape=(0, effective_layers, hidden_dim),
                    chunks=(chunk_t, eff_chunk_l, eff_chunk_h),
                    dtype=hs_dtype,
                    compressor=compressor,
                )

            if self.capture_spec.save_mean_states:
                self._arrays["hs_mean"] = hs_grp.require_dataset(
                    "mean",
                    shape=(n_samples, effective_layers, hidden_dim),
                    chunks=(min(64, n_samples), effective_layers, hidden_dim),
                    dtype="float32",
                    compressor=compressor,
                )

            if self.capture_spec.save_prompt_last:
                self._arrays["hs_prompt_last"] = hs_grp.require_dataset(
                    "prompt_last",
                    shape=(n_samples, effective_layers, hidden_dim),
                    chunks=(min(64, n_samples), effective_layers, hidden_dim),
                    dtype="float32",
                    compressor=compressor,
                )

        # -- Root-level metadata -------------------------------------------
        self._root.attrs["format_version"] = self.storage_spec.format_version
        self._root.attrs["openact_version"] = self.manifest.environment.openact_version

    # -- Worker error propagation ------------------------------------------

    def _set_worker_error(self, error: BaseException) -> None:
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = error

    def _check_worker_health(self) -> None:
        """Raise immediately if the background worker has died."""
        with self._worker_error_lock:
            err = self._worker_error
        if err is not None:
            raise RuntimeError(
                "AsyncZarrWriter worker failed (e.g. disk full or corrupt data). "
                "Some samples may not have been written."
            ) from err

    def _drain_queue(self) -> None:
        """Empty the queue after the worker has died to prevent caller deadlocks."""
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                drained += 1
            except queue.Empty:
                break
        if drained > 0:
            import logging

            logging.getLogger("openact.collect").warning(
                "Drained %d tasks from writer queue after worker error", drained
            )

    # -- Capacity check ----------------------------------------------------

    def _ensure_sample_capacity(self, sample_idx: int) -> None:
        if sample_idx >= self._allocated_samples:
            raise IndexError(
                f"sample_idx {sample_idx} >= allocated {self._allocated_samples}. "
                "Ensure manifest.dataset.n_samples matches the number of items "
                "from task.iter_items() (CollectionRunner sets this from len(plan))."
            )

    # -- Submission --------------------------------------------------------

    def submit(self, task: WriteTask) -> None:
        """Put a write task on the queue, checking worker health first."""
        self._check_worker_health()
        # Retry loop: if the queue is full we keep trying, but also check
        # for worker death so we never block forever.
        while True:
            self._check_worker_health()
            try:
                self._queue.put(task, timeout=1.0)
                return
            except queue.Full:
                continue

    def submit_sample(
        self,
        sample_idx: int,
        token_ids: np.ndarray,
        token_offsets: np.ndarray,
        hidden_state_data: Optional[Any] = None,
        status: SampleStatus = SampleStatus.OK,
        attention: Optional[Dict[int, np.ndarray]] = None,
        mlp: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        self.submit(
            WriteTask(
                task_type=WriteTaskType.SAMPLE,
                data={
                    "sample_idx": sample_idx,
                    "token_ids": token_ids,
                    "token_offsets": token_offsets,
                    "hidden_state_data": hidden_state_data,
                    "status": status,
                    "attention": attention,
                    "mlp": mlp,
                },
            )
        )

    def mark_failed(self, sample_idx: int, status: SampleStatus) -> None:
        self.submit(
            WriteTask(
                task_type=WriteTaskType.SAMPLE,
                data={
                    "sample_idx": sample_idx,
                    "token_ids": np.array([], dtype=np.int32),
                    "token_offsets": np.zeros((0, 2), dtype=np.int32),
                    "hidden_state_data": None,
                    "status": status,
                },
            )
        )

    # -- Background write loop ---------------------------------------------

    def _write_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    task = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if task.task_type == WriteTaskType.FINALIZE:
                        break
                    elif task.task_type == WriteTaskType.FLUSH:
                        pass  # Zarr auto-flushes on write
                    elif task.task_type == WriteTaskType.SAMPLE:
                        self._write_sample(task.data)
                except Exception:
                    # Re-raise so the outer handler can capture it
                    raise
                finally:
                    self._queue.task_done()
        except Exception as e:
            import logging

            logging.getLogger("openact.collect").error(
                "AsyncZarrWriter worker error: %s\n%s", e, traceback.format_exc()
            )
            self._set_worker_error(e)
            # Drain remaining tasks so callers waiting on queue.join() don't
            # deadlock.
            self._drain_queue()

    def _write_sample(self, data: Dict[str, Any]) -> None:
        sample_idx = data["sample_idx"]
        token_ids = data["token_ids"]
        token_offsets = data["token_offsets"]
        hidden_state_data = data.get("hidden_state_data")
        status = data.get("status", SampleStatus.OK)

        with self._lock:
            self._ensure_sample_capacity(sample_idx)
            n_tokens = len(token_ids)
            ptr_start = self._current_token_ptr

            if n_tokens > 0:
                new_size = ptr_start + n_tokens

                self._arrays["token_ids"].resize((new_size,))
                self._arrays["token_ids"][ptr_start:new_size] = token_ids

                self._arrays["token_offsets"].resize((new_size, 2))
                self._arrays["token_offsets"][ptr_start:new_size] = token_offsets

                if (
                    hidden_state_data is not None
                    and hidden_state_data.per_token_states is not None
                ):
                    if "hs_per_token" in self._arrays:
                        L = hidden_state_data.per_token_states.shape[1]
                        H = hidden_state_data.per_token_states.shape[2]
                        self._arrays["hs_per_token"].resize((new_size, L, H))
                        self._arrays["hs_per_token"][ptr_start:new_size] = (
                            hidden_state_data.per_token_states
                        )

                self._current_token_ptr = new_size

            ptr_end = self._current_token_ptr
            self._arrays["sample_ptr"][sample_idx] = ptr_start
            self._arrays["sample_ptr"][sample_idx + 1] = ptr_end
            self._arrays["sample_status"][sample_idx] = int(status)

            if hidden_state_data is not None:
                if hidden_state_data.mean_states is not None and "hs_mean" in self._arrays:
                    self._arrays["hs_mean"][sample_idx] = hidden_state_data.mean_states
                if (
                    hidden_state_data.prompt_last_states is not None
                    and "hs_prompt_last" in self._arrays
                ):
                    self._arrays["hs_prompt_last"][sample_idx] = (
                        hidden_state_data.prompt_last_states
                    )

    # -- Flush / finalize --------------------------------------------------

    def flush(self) -> None:
        self.submit(WriteTask(task_type=WriteTaskType.FLUSH))
        self._queue.join()

    def finalize(self) -> None:
        if not self._initialized:
            return
        self._check_worker_health()

        self.submit(WriteTask(task_type=WriteTaskType.FINALIZE))
        self._queue.join()

        if self._worker is not None:
            self._worker.join(timeout=30)
            if self._worker.is_alive():
                import logging

                logging.getLogger("openact.collect").error(
                    "AsyncZarrWriter worker did not terminate within 30s"
                )
                self._drain_queue()

        # Surface any error that happened in the worker
        with self._worker_error_lock:
            err = self._worker_error
        if err is not None:
            self._worker_error = None
            self._initialized = False
            raise RuntimeError(
                "AsyncZarrWriter worker failed (e.g. disk full or corrupt data). "
                "Some samples may not have been written."
            ) from err

        try:
            zarr.consolidate_metadata(str(self.zarr_path))
        except Exception:
            pass

        self._initialized = False

    # -- Properties --------------------------------------------------------

    @property
    def current_token_count(self) -> int:
        with self._lock:
            return self._current_token_ptr

    @property
    def pending_count(self) -> int:
        """Current number of pending tasks in the write queue."""
        return self._queue.qsize()

    def __repr__(self) -> str:
        status = "running" if self._initialized else "stopped"
        return f"AsyncZarrWriter(path='{self.zarr_path}', {status})"