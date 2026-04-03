from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import json


@dataclass
class ModelSpec:
    name: str = ''
    source: str = ''
    identifier: str = ''
    revision: Optional[str] = None
    dtype: str = 'bfloat16'
    architecture: Optional[str] = None
    n_layers: Optional[int] = None
    n_decoder_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    n_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    tokenizer_class: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelSpec':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class DatasetSpec:
    name: str = ''
    source: str = ''
    identifier: Optional[str] = None
    split: str = 'test'
    language: str = 'en'
    n_samples: Optional[int] = None
    max_samples: Optional[int] = None
    subset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSpec':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class PromptSpec:
    template_name: str = ''
    template_text: str = ''
    template_hash: str = ''
    variables: List[str] = field(default_factory=list)
    chat_template_applied: bool = True
    system_message: Optional[str] = None
    language: str = 'en'
    requested_language: str = 'en'
    localization_mode: str = 'base'
    answer_prefix: str = 'Answer'
    template_variant: Optional[str] = None
    supports_multilingual: bool = False

    def __post_init__(self) -> None:
        if not self.template_hash and self.template_text:
            self.template_hash = hashlib.sha256(self.template_text.encode('utf-8')).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptSpec':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class EnvironmentSpec:
    openact_version: str = ''
    python_version: str = ''
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    cuda_version: Optional[str] = None
    hostname: Optional[str] = None
    command_line: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentSpec':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class StatsSpec:
    n_samples_total: int = 0
    n_samples_ok: int = 0
    n_samples_error: int = 0
    n_samples_skipped: int = 0
    n_tokens_total: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatsSpec':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class CaptureConfig:
    hidden_states: bool = True
    hidden_states_layers: Optional[List[int]] = None
    hidden_states_dtype: str = 'float16'
    save_per_token: bool = True
    save_mean_states: bool = True
    save_prompt_last: bool = True
    compute_online_metrics: bool = True
    attention: bool = False
    attention_layers: Optional[List[int]] = None
    mlp: bool = False
    mlp_layers: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaptureConfig':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    do_sample: bool = False
    seed: int = 42
    stop_sequences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfig':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class StorageConfig:
    format_version: str = '1.0.0'
    compression: str = 'zstd'
    compression_level: int = 5
    chunk_tokens: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageConfig':
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class Manifest:
    schema_version: str = '1.0.0'
    run_id: str = ''
    created_at: str = ''
    model: ModelSpec = field(default_factory=ModelSpec)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    stats: StatsSpec = field(default_factory=StatsSpec)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    capture_config: Dict[str, Any] = field(default_factory=dict)
    storage_config: Dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + 'Z'
        if self.generation_config:
            self.generation = GenerationConfig.from_dict(self.generation_config)
        if self.capture_config:
            self.capture = CaptureConfig.from_dict(self.capture_config)
        if self.storage_config:
            self.storage = StorageConfig.from_dict(self.storage_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'run_id': self.run_id,
            'created_at': self.created_at,
            'model': self.model.to_dict(),
            'dataset': self.dataset.to_dict(),
            'prompt': self.prompt.to_dict(),
            'generation': self.generation_config or self.generation.to_dict(),
            'capture': self.capture_config or self.capture.to_dict(),
            'storage': self.storage_config or self.storage.to_dict(),
            'environment': self.environment.to_dict(),
            'stats': self.stats.to_dict(),
            'custom': self.custom,
        }

    def save(self, path: Union[str, Path]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Manifest':
        input_path = Path(path)
        with open(input_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        generation_data = data.get('generation', {})
        capture_data = data.get('capture', {})
        storage_data = data.get('storage', {})
        if not isinstance(generation_data, dict):
            generation_data = {}
        if not isinstance(capture_data, dict):
            capture_data = {}
        if not isinstance(storage_data, dict):
            storage_data = {}
        return cls(
            schema_version=data.get('schema_version', '1.0.0'),
            run_id=data.get('run_id', ''),
            created_at=data.get('created_at', ''),
            model=ModelSpec.from_dict(data.get('model', {})),
            dataset=DatasetSpec.from_dict(data.get('dataset', {})),
            prompt=PromptSpec.from_dict(data.get('prompt', {})),
            generation_config=generation_data,
            capture_config=capture_data,
            storage_config=storage_data,
            environment=EnvironmentSpec.from_dict(data.get('environment', {})),
            stats=StatsSpec.from_dict(data.get('stats', {})),
            custom=data.get('custom', {}),
        )

    def validate(self) -> List[str]:
        issues: List[str] = []
        if not self.run_id:
            issues.append('Missing run_id')
        if not self.model.name:
            issues.append('Missing model.name')
        if not self.model.identifier:
            issues.append('Missing model.identifier')
        if not self.dataset.name:
            issues.append('Missing dataset.name')
        if self.model.n_layers is None:
            issues.append('Missing model.n_layers')
        if self.model.hidden_dim is None:
            issues.append('Missing model.hidden_dim')
        return issues
