from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import zarr
from numcodecs import Blosc
from openact_collect.schema import CaptureSpec, StorageSpec
from openact_core.schema.manifest import Manifest
from openact_core.schema.status import SampleStatus


class ZarrWriter:
    def __init__(self, zarr_path: Path, manifest: Manifest, capture_spec: CaptureSpec, storage_spec: StorageSpec, queue_size: int = 8):
        self.zarr_path = Path(zarr_path)
        self.manifest = manifest
        self.capture_spec = capture_spec
        self.storage_spec = storage_spec
        self.queue_size = queue_size
        self._root: Optional[zarr.Group] = None
        self._arrays: Dict[str, Any] = {}
        self._current_token_ptr = 0
        self._allocated_samples = 0
        self._initialized = False

    def __enter__(self) -> 'ZarrWriter':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.finalize()
        return False

    def start(self) -> None:
        if self._initialized:
            return
        self._initialize_zarr()
        self._initialized = True

    def _initialize_zarr(self) -> None:
        if self.zarr_path.exists():
            raise FileExistsError(f'Refusing to overwrite existing tensors: {self.zarr_path}')
        self.zarr_path.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname=self.storage_spec.compression, clevel=self.storage_spec.compression_level)
        self._root = zarr.open_group(str(self.zarr_path), mode='w')
        n_samples = int(self.manifest.dataset.n_samples or 0)
        n_layers = int(self.manifest.model.n_layers or 0)
        n_decoder_layers = int(getattr(self.manifest.model, 'n_decoder_layers', 0) or max(0, n_layers - 1))
        hidden_dim = int(self.manifest.model.hidden_dim or 0)
        layers = self.capture_spec.get_effective_layers(n_layers) if self.capture_spec.hidden_states else []
        effective_layers = len(layers)
        self._allocated_samples = n_samples
        chunk_t = max(1, int(self.storage_spec.chunk_tokens))
        chunk_l = max(1, min(int(self.storage_spec.chunk_layers), max(1, effective_layers)))
        chunk_h = max(1, min(int(self.storage_spec.chunk_hidden_dim), max(1, hidden_dim)))
        tokens = self._root.require_group('tokens')
        self._arrays['token_ids'] = tokens.require_dataset('ids', shape=(0,), chunks=(chunk_t * 16,), dtype='int32', compressor=compressor)
        self._arrays['token_offsets'] = tokens.require_dataset('offsets', shape=(0, 2), chunks=(chunk_t * 16, 2), dtype='int32', compressor=compressor)
        self._arrays['sample_ptr'] = tokens.require_dataset('sample_ptr', shape=(n_samples + 1,), chunks=(min(4096, n_samples + 1),), dtype='int64', fill_value=-1)
        self._arrays['sample_ptr'][0] = 0
        self._arrays['sample_status'] = self._root.require_dataset('sample_status', shape=(n_samples,), chunks=(min(4096, max(1, n_samples)),), dtype='int8', fill_value=int(SampleStatus.UNPROCESSED))
        if self.capture_spec.hidden_states:
            hidden_states = self._root.require_group('hidden_states')
            hidden_states.attrs['layers'] = layers
            hidden_states.attrs['dtype'] = self.capture_spec.hidden_states_dtype
            hidden_states.attrs['token_alignment'] = 'response_token'
            hidden_states.attrs['layer_semantics'] = 'HuggingFace hidden_states indices; last entry is final norm output'
            hs_dtype = self.capture_spec.hidden_states_dtype
            if self.capture_spec.save_per_token:
                self._arrays['hs_per_token'] = hidden_states.require_dataset('per_token', shape=(0, effective_layers, hidden_dim), chunks=(chunk_t, chunk_l, chunk_h), dtype=hs_dtype, compressor=compressor)
            if self.capture_spec.save_mean_states:
                # Samples are written individually. A 64-row all-layer chunk
                # would repeatedly read/recompress tens of MB for each new row.
                self._arrays['hs_mean'] = hidden_states.require_dataset('mean', shape=(n_samples, effective_layers, hidden_dim), chunks=(1, max(1, effective_layers), max(1, hidden_dim)), dtype='float32', compressor=compressor)
            if self.capture_spec.save_prompt_last:
                self._arrays['hs_prompt_last'] = hidden_states.require_dataset('prompt_last', shape=(n_samples, effective_layers, hidden_dim), chunks=(1, max(1, effective_layers), max(1, hidden_dim)), dtype='float32', compressor=compressor)
            if self.capture_spec.final_norm:
                norm = self._root.require_group('final_norm')
                norm.attrs['token_alignment'] = 'response_token'
                norm.attrs['semantics'] = 'input and output of the final normalization module'
                for side in ('pre', 'post'):
                    group = norm.require_group(side)
                    if self.capture_spec.save_per_token:
                        self._arrays[f'final_norm_{side}'] = group.create_dataset('per_token', shape=(0, hidden_dim), chunks=(chunk_t, chunk_h), dtype=hs_dtype, compressor=compressor)
                    for enabled, suffix, name in (
                        (self.capture_spec.save_mean_states, 'mean', 'mean'),
                        (self.capture_spec.save_prompt_last, 'prompt_last', 'prompt_last'),
                    ):
                        if enabled:
                            self._arrays[f'final_norm_{side}_{suffix}'] = group.create_dataset(name, shape=(n_samples, hidden_dim), chunks=(min(64, max(1, n_samples)), hidden_dim), dtype='float32', compressor=compressor)
        if self.capture_spec.attention and (self.capture_spec.attention_save_outputs or self.capture_spec.attention_save_patterns):
            attention = self._root.require_group('attention')
            attention_layers = self.capture_spec.get_effective_attention_layers(n_decoder_layers)
            attention.attrs['layers'] = attention_layers
            attention.attrs['pattern_window'] = int(getattr(self.capture_spec, 'attention_pattern_window', 256) or 256)
            if self.capture_spec.attention_save_outputs:
                self._arrays['attn_output'] = attention.require_dataset('output', shape=(0, len(attention_layers), hidden_dim), chunks=(chunk_t, max(1, len(attention_layers)), chunk_h), dtype='float16', compressor=compressor)
            if self.capture_spec.attention_save_patterns:
                n_heads = int(getattr(self.manifest.model, 'n_heads', 0) or 0)
                window = int(getattr(self.capture_spec, 'attention_pattern_window', 256) or 256)
                if n_heads > 0:
                    self._arrays['attn_pattern'] = attention.require_dataset('pattern', shape=(0, len(attention_layers), n_heads, window), chunks=(min(256, chunk_t), max(1, len(attention_layers)), 1, window), dtype='float16', compressor=compressor)
        if self.capture_spec.mlp and self.capture_spec.mlp_save_output:
            mlp = self._root.require_group('mlp')
            mlp_layers = self.capture_spec.get_effective_mlp_layers(n_decoder_layers)
            mlp.attrs['layers'] = mlp_layers
            self._arrays['mlp_output'] = mlp.require_dataset('output', shape=(0, len(mlp_layers), hidden_dim), chunks=(chunk_t, max(1, len(mlp_layers)), chunk_h), dtype='float16', compressor=compressor)
        self._root.attrs['format_version'] = self.storage_spec.format_version
        self._root.attrs['openact_version'] = self.manifest.environment.openact_version

    def _ensure_sample_capacity(self, sample_idx: int) -> None:
        if sample_idx >= self._allocated_samples:
            raise IndexError(f'sample_idx {sample_idx} >= allocated {self._allocated_samples}')

    def submit_sample(self, sample_idx: int, token_ids: np.ndarray, token_offsets: np.ndarray, hidden_state_data: Optional[Any] = None, status: SampleStatus = SampleStatus.OK, attention: Optional[Dict[int, np.ndarray]] = None, mlp: Optional[Dict[int, np.ndarray]] = None) -> None:
        del attention, mlp
        self._write_sample(sample_idx=sample_idx, token_ids=token_ids, token_offsets=token_offsets, hidden_state_data=hidden_state_data, status=status)

    def mark_failed(self, sample_idx: int, status: SampleStatus) -> None:
        self._write_sample(sample_idx=sample_idx, token_ids=np.array([], dtype=np.int32), token_offsets=np.zeros((0, 2), dtype=np.int32), hidden_state_data=None, status=status)

    def _write_sample(self, sample_idx: int, token_ids: np.ndarray, token_offsets: np.ndarray, hidden_state_data: Optional[Any], status: SampleStatus) -> None:
        if not self._initialized:
            self.start()
        self._ensure_sample_capacity(sample_idx)
        n_tokens = int(len(token_ids))
        if token_offsets.shape != (n_tokens, 2):
            raise ValueError('Token offsets must have one (start, end) pair per token')
        if hidden_state_data is not None:
            for key, field in (
                ('hs_per_token', 'per_token_states'), ('attn_output', 'attention_outputs'),
                ('attn_pattern', 'attention_patterns'), ('mlp_output', 'mlp_outputs'),
                ('final_norm_pre', 'final_norm_pre'), ('final_norm_post', 'final_norm_post'),
            ):
                array = self._arrays.get(key)
                if array is not None:
                    value = getattr(hidden_state_data, field, None)
                    expected = (n_tokens, *array.shape[1:])
                    if value is None or value.shape != expected:
                        raise ValueError(f'{field} must have shape {expected}; got {getattr(value, "shape", None)}')
        ptr_start = self._current_token_ptr
        ptr_end = ptr_start + n_tokens
        if n_tokens:
            self._arrays['token_ids'].resize((ptr_end,))
            self._arrays['token_ids'][ptr_start:ptr_end] = token_ids
            self._arrays['token_offsets'].resize((ptr_end, 2))
            self._arrays['token_offsets'][ptr_start:ptr_end] = token_offsets
            if hidden_state_data is not None and hidden_state_data.per_token_states is not None:
                hs_per_token = self._arrays.get('hs_per_token')
                if hs_per_token is not None:
                    n_layers = hidden_state_data.per_token_states.shape[1]
                    hidden_dim = hidden_state_data.per_token_states.shape[2]
                    hs_per_token.resize((ptr_end, n_layers, hidden_dim))
                    hs_per_token[ptr_start:ptr_end] = hidden_state_data.per_token_states
            if hidden_state_data is not None and hidden_state_data.attention_outputs is not None:
                attention_output = self._arrays.get('attn_output')
                if attention_output is not None:
                    n_layers = hidden_state_data.attention_outputs.shape[1]
                    hidden_dim = hidden_state_data.attention_outputs.shape[2]
                    attention_output.resize((ptr_end, n_layers, hidden_dim))
                    attention_output[ptr_start:ptr_end] = hidden_state_data.attention_outputs
            if hidden_state_data is not None and hidden_state_data.attention_patterns is not None:
                attention_pattern = self._arrays.get('attn_pattern')
                if attention_pattern is not None:
                    n_layers = hidden_state_data.attention_patterns.shape[1]
                    n_heads = hidden_state_data.attention_patterns.shape[2]
                    window = hidden_state_data.attention_patterns.shape[3]
                    attention_pattern.resize((ptr_end, n_layers, n_heads, window))
                    attention_pattern[ptr_start:ptr_end] = hidden_state_data.attention_patterns
            if hidden_state_data is not None and hidden_state_data.mlp_outputs is not None:
                mlp_output = self._arrays.get('mlp_output')
                if mlp_output is not None:
                    n_layers = hidden_state_data.mlp_outputs.shape[1]
                    hidden_dim = hidden_state_data.mlp_outputs.shape[2]
                    mlp_output.resize((ptr_end, n_layers, hidden_dim))
                    mlp_output[ptr_start:ptr_end] = hidden_state_data.mlp_outputs
        self._current_token_ptr = ptr_end
        self._arrays['sample_ptr'][sample_idx] = ptr_start
        self._arrays['sample_ptr'][sample_idx + 1] = ptr_end
        self._arrays['sample_status'][sample_idx] = int(status)
        if hidden_state_data is None:
            return
        for side in ('pre', 'post'):
            key = f'final_norm_{side}'
            if key in self._arrays:
                array = self._arrays[key]
                array.resize((ptr_end, array.shape[1]))
                array[ptr_start:ptr_end] = getattr(hidden_state_data, key)
            for suffix in ('mean', 'prompt_last'):
                key = f'final_norm_{side}_{suffix}'
                if key in self._arrays:
                    self._arrays[key][sample_idx] = getattr(hidden_state_data, key)
        if hidden_state_data.mean_states is not None and 'hs_mean' in self._arrays:
            self._arrays['hs_mean'][sample_idx] = hidden_state_data.mean_states
        if hidden_state_data.prompt_last_states is not None and 'hs_prompt_last' in self._arrays:
            self._arrays['hs_prompt_last'][sample_idx] = hidden_state_data.prompt_last_states

    def flush(self) -> None:
        return None

    def finalize(self) -> None:
        if not self._initialized:
            return
        sample_ptr = self._arrays.get('sample_ptr')
        if sample_ptr is not None:
            # One bulk read, rather than one NFS chunk read per sample.
            values = sample_ptr[:]
            missing = values < 0
            if np.any(missing):
                previous = np.maximum.accumulate(np.where(~missing, np.arange(len(values)), 0))
                values[missing] = values[previous[missing]]
                sample_ptr[:] = values
        try:
            zarr.consolidate_metadata(str(self.zarr_path))
        except Exception:
            pass
        self._initialized = False

    @property
    def current_token_count(self) -> int:
        return self._current_token_ptr

    @property
    def pending_count(self) -> int:
        return 0

    def __repr__(self) -> str:
        state = 'running' if self._initialized else 'stopped'
        return f'ZarrWriter(path={str(self.zarr_path)!r}, {state})'


AsyncZarrWriter = ZarrWriter
