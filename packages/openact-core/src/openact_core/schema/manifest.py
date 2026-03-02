from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json
import hashlib
from datetime import datetime


@dataclass
class ModelSpec:
    name: str = ""
    source: str = ""
    identifier: str = ""
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    architecture: Optional[str] = None
    n_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    n_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    tokenizer_class: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetSpec:
    name: str = ""
    source: str = ""
    identifier: Optional[str] = None
    split: str = "test"
    language: str = "en"
    n_samples: Optional[int] = None
    max_samples: Optional[int] = None
    subset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PromptSpec:
    template_name: str = ""
    template_text: str = ""
    template_hash: str = ""
    variables: List[str] = field(default_factory=list)
    chat_template_applied: bool = True
    system_message: Optional[str] = None

    def __post_init__(self):
        if not self.template_hash and self.template_text:
            self.template_hash = hashlib.sha256(
                self.template_text.encode("utf-8")
            ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EnvironmentSpec:
    openact_version: str = ""
    python_version: str = ""
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    cuda_version: Optional[str] = None
    hostname: Optional[str] = None
    command_line: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
    def from_dict(cls, d: Dict[str, Any]) -> "StatsSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Typed config dataclasses for capture / generation / storage.
# These mirror the collect-side specs but live in core so that readers
# can deserialize them without depending on openact-collect.
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Typed representation of capture settings stored in manifest."""

    hidden_states: bool = True
    hidden_states_layers: Optional[List[int]] = None
    hidden_states_dtype: str = "float16"
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
    def from_dict(cls, d: Dict[str, Any]) -> "CaptureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GenerationConfig:
    """Typed representation of generation settings stored in manifest."""

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
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StorageConfig:
    """Typed representation of storage settings stored in manifest."""

    format_version: str = "1.0.0"
    compression: str = "zstd"
    compression_level: int = 5
    chunk_tokens: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StorageConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    schema_version: str = "1.0.0"
    run_id: str = ""
    created_at: str = ""
    model: ModelSpec = field(default_factory=ModelSpec)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    stats: StatsSpec = field(default_factory=StatsSpec)

    # Raw dicts kept for backward compat and passthrough of unknown keys.
    generation_config: Dict[str, Any] = field(default_factory=dict)
    capture_config: Dict[str, Any] = field(default_factory=dict)
    storage_config: Dict[str, Any] = field(default_factory=dict)

    # Typed views — populated from the raw dicts on load.
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        # Sync typed views from raw dicts if they were populated.
        if self.generation_config:
            self.generation = GenerationConfig.from_dict(self.generation_config)
        if self.capture_config:
            self.capture = CaptureConfig.from_dict(self.capture_config)
        if self.storage_config:
            self.storage = StorageConfig.from_dict(self.storage_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "prompt": self.prompt.to_dict(),
            "generation": self.generation_config or self.generation.to_dict(),
            "capture": self.capture_config or self.capture.to_dict(),
            "storage": self.storage_config or self.storage.to_dict(),
            "environment": self.environment.to_dict(),
            "stats": self.stats.to_dict(),
            "custom": self.custom,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Manifest":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gen_data = data.get("generation", {})
        capture_data = data.get("capture", {})
        storage_data = data.get("storage", {})
        if not isinstance(gen_data, dict):
            gen_data = {}
        if not isinstance(capture_data, dict):
            capture_data = {}
        if not isinstance(storage_data, dict):
            storage_data = {}

        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            run_id=data.get("run_id", ""),
            created_at=data.get("created_at", ""),
            model=ModelSpec.from_dict(data.get("model", {})),
            dataset=DatasetSpec.from_dict(data.get("dataset", {})),
            prompt=PromptSpec.from_dict(data.get("prompt", {})),
            generation_config=gen_data,
            capture_config=capture_data,
            storage_config=storage_data,
            environment=EnvironmentSpec.from_dict(data.get("environment", {})),
            stats=StatsSpec.from_dict(data.get("stats", {})),
            custom=data.get("custom", {}),
        )

    def validate(self) -> List[str]:
        issues = []
        if not self.run_id:
            issues.append("Missing run_id")
        if not self.model.name:
            issues.append("Missing model.name")
        if not self.model.identifier:
            issues.append("Missing model.identifier")
        if not self.dataset.name:
            issues.append("Missing dataset.name")
        if self.model.n_layers is None:
            issues.append("Missing model.n_layers")
        if self.model.hidden_dim is None:
            issues.append("Missing model.hidden_dim")
        return issues