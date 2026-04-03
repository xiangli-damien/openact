"""Dataset utilities for OpenAct.

This package exists to keep *dataset acquisition* (download / caching) and
*prompt rendering* separated from the actual collection loop.

Tasks should avoid calling Hugging Face `datasets.load_dataset` directly.
Instead, they should use :func:`openact_collect.data.load_hf_dataset`.
"""

from openact_collect.data.hf import HFDatasetSpec, load_hf_dataset, prepare_hf_dataset
from openact_collect.data.prepared import (
    PreparedParquetDataset,
    PreparedDatasetWriter,
    write_prepared_parquet,
)

__all__ = [
    "HFDatasetSpec",
    "load_hf_dataset",
    "prepare_hf_dataset",
    "PreparedParquetDataset",
    "PreparedDatasetWriter",
    "write_prepared_parquet",
]
