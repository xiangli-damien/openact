from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class StorageSpec:
    format_version: str = '1.0.0'
    compression: str = 'zstd'
    compression_level: int = 5
    chunk_tokens: int = 256
    chunk_layers: int = 4
    chunk_hidden_dim: int = 512
    store_token_offsets: bool = True
    zarr_store_type: str = 'directory'

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StorageSpec':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
