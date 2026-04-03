from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        item = getattr(value, "item", None)
        if callable(item):
            scalar = item()
            if scalar is None or isinstance(scalar, (str, int, float, bool)):
                return scalar
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    return str(value)


def normalize_row_for_parquet(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        out[str(k)] = _normalize_value(v)
    return out


@dataclass
class PreparedParquetDataset:
    path: Union[str, Path]

    def iter_rows(self, max_rows: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        df = pd.read_parquet(self.path)
        if max_rows is not None:
            df = df.iloc[: int(max_rows)]
        for _idx, row in df.iterrows():
            yield row.to_dict()


class PreparedDatasetWriter:
    def __init__(self, out_dir: Union[str, Path], batch_rows: int = 5000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_rows = int(batch_rows)
        self._buffer: List[Dict[str, Any]] = []
        self._part_idx = 0

    def add(self, row: Dict[str, Any]) -> None:
        self._buffer.append(normalize_row_for_parquet(row))
        if len(self._buffer) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        part_path = self.out_dir / f"part_{self._part_idx:05d}.parquet"
        pd.DataFrame(self._buffer).to_parquet(part_path, index=False)
        self._buffer.clear()
        self._part_idx += 1

    def close(self) -> None:
        self.flush()


def write_prepared_parquet(rows: Sequence[Dict[str, Any]], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized: List[Dict[str, Any]] = []
    for r in rows:
        normalized.append(normalize_row_for_parquet(dict(r)))

    pd.DataFrame(normalized).to_parquet(out_path, index=False)
    return out_path