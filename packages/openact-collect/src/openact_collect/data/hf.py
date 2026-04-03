"""Hugging Face dataset loading helpers.

Why this exists:
  - Keep dataset download/caching in one place.
  - Make tasks small and readable.
  - Enable an explicit "prepare dataset" stage independent of model execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class HFDatasetSpec:
    """A minimal description of a Hugging Face dataset."""

    name: str
    config: Optional[str] = None
    split: str = "test"
    revision: Optional[str] = None
    streaming: bool = False
    cache_dir: Optional[str] = None


def load_hf_dataset(spec: HFDatasetSpec) -> Any:
    """Load a Hugging Face dataset according to the provided spec."""

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required to load Hugging Face datasets. "
            "Install it with: pip install datasets"
        ) from exc

    kwargs = {
        "split": spec.split,
        "streaming": spec.streaming,
    }
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    if spec.cache_dir is not None:
        kwargs["cache_dir"] = spec.cache_dir

    if spec.config is None:
        return load_dataset(spec.name, **kwargs)
    return load_dataset(spec.name, spec.config, **kwargs)


def prepare_hf_dataset(spec: HFDatasetSpec) -> None:
    """Eagerly download/cache a dataset.

    This is intentionally a thin wrapper: it simply loads the dataset once so
    Hugging Face does its standard caching. The returned dataset object is
    discarded.
    """

    _ = load_hf_dataset(spec)
