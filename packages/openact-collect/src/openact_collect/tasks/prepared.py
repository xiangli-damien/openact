from __future__ import annotations
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union
import pandas as pd
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry
from openact_core.tasks.templates import PromptTemplate


@TaskRegistry.register('prepared')
class PreparedParquetTask(Task):
    source = 'prepared_parquet'

    def __init__(self, prepared_path: Union[str, Path], max_samples: Optional[int] = None, **kwargs: Any):
        # Prepared rows already contain rendered prompts and their own split.
        kwargs.pop('split', None)
        kwargs.pop('template', None)
        super().__init__(max_samples=max_samples, split='', template=None, **kwargs)
        self.prepared_path = Path(prepared_path)
        self._files = self._resolve_files(self.prepared_path)
        self._prepared_manifest = self._load_prepared_manifest()
        self._apply_prepared_manifest()

    @staticmethod
    def _resolve_files(path: Path):
        if path.is_dir():
            files = sorted(glob.glob(str(path / 'part_*.parquet')))
            if not files:
                files = sorted(glob.glob(str(path / '*.parquet')))
            return [Path(p) for p in files]
        return [path]

    def _resolve_manifest_path(self) -> Optional[Path]:
        base_dir = self.prepared_path if self.prepared_path.is_dir() else self.prepared_path.parent
        manifest_path = base_dir / 'prepared_manifest.json'
        return manifest_path if manifest_path.exists() else None

    def _load_prepared_manifest(self) -> Dict[str, Any]:
        manifest_path = self._resolve_manifest_path()
        if manifest_path is None:
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _apply_prepared_manifest(self) -> None:
        if not self._prepared_manifest:
            return
        self.dataset_sources = self._prepared_manifest.get('dataset_sources', [])
        task_name = self._prepared_manifest.get('task')
        if task_name:
            self.task_name = str(task_name)
        task_source = self._prepared_manifest.get('task_source')
        if task_source:
            self.source = str(task_source)
        split = self._prepared_manifest.get('split')
        if split is not None:
            self.split = str(split)
        language = self._prepared_manifest.get('language')
        if language:
            self.language = str(language)
        task_type = self._prepared_manifest.get('task_type')
        if task_type:
            self.task_type = str(task_type)
        template_variant = self._prepared_manifest.get('prompt_template_variant') or self._prepared_manifest.get('template_variant')
        if template_variant:
            self._template_variant = str(template_variant)

    def estimate_size(self) -> Optional[int]:
        if len(self._files) != 1:
            return None
        try:
            df = pd.read_parquet(self._files[0], columns=['prompt_text'])
        except Exception:
            return None
        valid = df['prompt_text'].fillna('').astype(str).str.len() > 0
        n = int(valid.sum())
        return min(n, self.max_samples) if self.max_samples is not None else n

    @staticmethod
    def _parse_prompt_fields(raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def get_prompt_template_for_item(self, item: Optional[TaskItem] = None) -> PromptTemplate:
        template_data = self._prepared_manifest.get('prompt_template')
        if isinstance(template_data, dict) and template_data:
            try:
                return PromptTemplate.from_dict(template_data)
            except Exception:
                pass
        return PromptTemplate(name='prepared_raw', template='{prompt}', variables=('prompt',), answer_prefix='')

    def iter_items(self) -> Iterator[TaskItem]:
        idx = 0
        for fp in self._files:
            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                if self.max_samples is not None and idx >= self.max_samples:
                    return
                record: Dict[str, Any] = row.to_dict()
                prompt_text = record.get('prompt_text')
                if pd.isna(prompt_text) or str(prompt_text) == '':
                    continue
                sample_id = record.get('sample_id', f'prepared_{idx}')
                ground_truth = record.get('ground_truth')
                language = record.get('language', self.language or 'en')
                prompt_fields = self._parse_prompt_fields(record.get('prompt_fields_json'))
                meta = {
                    key: value
                    for key, value in record.items()
                    if key not in {'sample_idx', 'sample_id', 'prompt_text', 'prompt_fields_json', 'ground_truth', 'language'}
                }
                yield TaskItem(
                    sample_idx=idx,
                    sample_id=str(sample_id),
                    prompt_text=str(prompt_text),
                    prompt_fields=prompt_fields,
                    ground_truth=None if pd.isna(ground_truth) else str(ground_truth),
                    language=str(language),
                    meta=meta,
                )
                idx += 1
