from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from openact_collect.engine.model_manager import ModelRunner
from openact_collect.tasks.base import Task, TaskItem


def _json_or_none(value: Any) -> Optional[str]:
    if value in (None, {}, [], ()):
        return None
    return json.dumps(value, ensure_ascii=False)


def build_prepared_row(task: Task, item: TaskItem, prompt_text: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        'sample_idx': int(item.sample_idx),
        'sample_id': str(item.sample_id),
        'prompt_text': str(prompt_text),
        'prompt_fields_json': _json_or_none(item.prompt_fields or {}),
        'ground_truth': item.ground_truth,
        'language': str(getattr(item, 'language', 'en')),
        'semantic_meta_json': _json_or_none(item.semantic_meta()),
        **task.get_task_metadata(item),
        **task.get_prompt_template_metadata(item),
    }
    for key, value in item.parquet_meta().items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        row.setdefault(key, value)
    return row


def build_item_preview(task: Task, item: TaskItem, model_runner: Optional[ModelRunner] = None) -> Dict[str, Any]:
    prompt_text = task.render_prompt(item)
    preview: Dict[str, Any] = {
        'sample_idx': item.sample_idx,
        'sample_id': item.sample_id,
        'language': item.language,
        'ground_truth': item.ground_truth,
        'prompt_fields': dict(item.prompt_fields or {}),
        'meta': dict(item.meta or {}),
        'semantic_meta': item.semantic_meta(),
        'task_metadata': task.get_task_metadata(item),
        'prompt_metadata': task.get_prompt_template_metadata(item),
        'rendered_prompt': prompt_text,
        'rendered_prompt_len': len(prompt_text),
        'prepared_row_preview': build_prepared_row(task, item, prompt_text),
    }
    if model_runner is not None:
        if not model_runner._loaded:
            model_runner.load()
        messages = [{'role': 'user', 'content': prompt_text}]
        input_ids = model_runner.apply_chat_template(messages)
        chat_text = model_runner.render_chat_text(messages)
        token_ids = input_ids[0].tolist() if hasattr(input_ids, 'ndim') and int(input_ids.ndim) == 2 else input_ids.tolist()
        preview['model_input'] = {
            'messages': messages,
            'chat_text': chat_text,
            'n_input_tokens': int(input_ids.shape[-1]),
            'input_token_ids_head': token_ids[:64],
            'input_token_ids_tail': token_ids[-64:] if len(token_ids) > 64 else token_ids,
        }
    return preview


def inspect_task(task: Task, n_items: int = 3, model_runner: Optional[ModelRunner] = None) -> Dict[str, Any]:
    items: List[TaskItem] = []
    for index, item in enumerate(task.iter_items()):
        if index >= n_items:
            break
        items.append(item)
    estimated_total: Optional[int]
    try:
        estimated_total = task.estimate_size()
    except Exception:
        estimated_total = None
    payload: Dict[str, Any] = {
        'task': {
            'task_name': task.name,
            'source': task.source,
            'split': task.split,
            'language': getattr(task, 'language', 'en'),
            'task_type': getattr(task, 'task_type', 'capability'),
            'default_template': getattr(task, 'default_template', None),
            'template_variant': getattr(task, '_template_variant', None),
            'max_samples': getattr(task, 'max_samples', None),
            'config': dict(getattr(task, '_config', {}) or {}),
        },
        'prompt_template': task.get_prompt_template().to_dict(),
        'estimated_total': estimated_total,
        'preview_count': len(items),
        'items': [build_item_preview(task, item, model_runner=model_runner) for item in items],
    }
    return payload
