from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
from openact_core.tasks.templates import PromptTemplate, get_template

_SEMANTIC_META_KEYS = {
    'answer_type',
    'attack_method',
    'behavior_id',
    'category',
    'concept',
    'correct_answer_num',
    'lang_code',
    'language',
    'level',
    'profile',
    'prompt_variant',
    'question_concept',
    'rep_idx',
    'split',
    'subject',
}
_TASK_META_KEYS = ('task_name', 'task_source', 'task_split', 'task_type')
_PROMPT_META_KEYS = (
    'prompt_template_name',
    'prompt_template_hash',
    'prompt_template_language',
    'prompt_requested_language',
    'prompt_localization_mode',
    'prompt_answer_prefix',
    'prompt_template_variant',
)


def _is_present(value: Any) -> bool:
    return value not in (None, '', [], {}, ())


@dataclass
class TaskItem:
    sample_idx: int
    sample_id: str
    prompt_text: Optional[str] = None
    prompt_fields: Dict[str, Any] = field(default_factory=dict)
    ground_truth: Optional[str] = None
    language: str = 'en'
    meta: Dict[str, Any] = field(default_factory=dict)

    def semantic_meta(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in sorted(_SEMANTIC_META_KEYS):
            if key in self.meta and _is_present(self.meta[key]):
                out[key] = self.meta[key]
        for attr in ('behavior_id', 'split', 'category', 'prompt_variant', 'attack_method', 'profile', 'rep_idx'):
            if hasattr(self, attr):
                value = getattr(self, attr)
                if _is_present(value):
                    out[attr] = value
        return out

    def parquet_meta(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'language': self.language}
        semantic_meta = self.semantic_meta()
        if semantic_meta:
            out['semantic_meta'] = semantic_meta
        for key, value in self.meta.items():
            if key not in ('sample_id', 'prompt_text', 'ground_truth'):
                out[key] = value
        return out


class Task(ABC):
    task_name: str = 'base_task'
    task_type: str = 'capability'
    source: str = 'unknown'
    split: str = 'test'
    language: str = 'en'
    max_samples: Optional[int] = None
    default_template: str = 'cot'

    def get_profiles(self) -> Dict[str, Any]:
        return {}

    def get_safety_spec(self) -> Optional[Any]:
        return None

    def __init__(self, max_samples: Optional[int] = None, split: str = 'test', template: Optional[str] = None, **kwargs: Any):
        if max_samples is not None:
            self.max_samples = max_samples
        self.split = split
        self._template_variant = template or self.default_template
        self._config = kwargs
        self.dataset_revision = kwargs.get('dataset_revision')
        self.dataset_sources: List[Dict[str, Any]] = []
        if max_samples is not None and max_samples < 1:
            raise ValueError('max_samples must be positive')

    def load_hf_dataset(self, spec):
        from dataclasses import asdict, replace
        from openact_collect.data.hf import DATASET_REVISIONS, load_hf_dataset
        spec = replace(spec, revision=self.dataset_revision or spec.revision or DATASET_REVISIONS.get(spec.name))
        dataset = load_hf_dataset(spec)
        source = asdict(spec)
        source['fingerprint'] = getattr(dataset, '_fingerprint', None)
        if source not in self.dataset_sources:
            self.dataset_sources.append(source)
        return dataset

    @abstractmethod
    def iter_items(self) -> Iterator[TaskItem]:
        ...

    def estimate_size(self) -> Optional[int]:
        return None

    def get_prompt_template_for_item(self, item: Optional[TaskItem] = None) -> PromptTemplate:
        language = getattr(item, 'language', None) or getattr(self, 'language', 'en')
        return get_template(self.task_name, self._template_variant, language=language)

    def get_prompt_template(self) -> PromptTemplate:
        return self.get_prompt_template_for_item(None)

    def get_task_metadata(self, item: Optional[TaskItem] = None) -> Dict[str, Any]:
        out = {
            'task_name': self.name,
            'task_source': self.source,
            'task_split': self.split,
            'task_type': self.task_type,
        }
        if item is not None:
            for key in _TASK_META_KEYS:
                value = item.meta.get(key)
                if _is_present(value):
                    out[key] = value
        return out

    def get_prompt_template_metadata(self, item: Optional[TaskItem] = None) -> Dict[str, Any]:
        if item is not None:
            existing = {key: item.meta[key] for key in _PROMPT_META_KEYS if _is_present(item.meta.get(key))}
            if len(existing) == len(_PROMPT_META_KEYS):
                return existing
        template = self.get_prompt_template_for_item(item)
        out = {
            'prompt_template_name': template.name,
            'prompt_template_hash': template.hash,
            'prompt_template_language': template.language,
            'prompt_requested_language': template.requested_language,
            'prompt_localization_mode': template.localization_mode,
            'prompt_answer_prefix': template.answer_prefix,
            'prompt_template_variant': self._template_variant,
        }
        if item is not None:
            for key in _PROMPT_META_KEYS:
                value = item.meta.get(key)
                if _is_present(value):
                    out[key] = value
        return out

    def render_prompt(self, item: TaskItem) -> str:
        if item.prompt_text is not None:
            return str(item.prompt_text)
        template = self.get_prompt_template_for_item(item)
        return template.format(**(item.prompt_fields or {}))

    @property
    def name(self) -> str:
        return self.task_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.task_name}', split='{self.split}', template='{self._template_variant}')"


class GenericTask(Task):
    task_name = 'generic'
    source = 'custom'

    def __init__(self, data: List[Dict[str, Any]], prompt_key: str = 'prompt', answer_key: Optional[str] = 'answer', id_key: Optional[str] = None, template_name: str = 'generic', template_text: str = '{prompt}', max_samples: Optional[int] = None, **kwargs: Any):
        super().__init__(max_samples=max_samples, **kwargs)
        self._data = data
        self._prompt_key = prompt_key
        self._answer_key = answer_key
        self._id_key = id_key
        self._template_name = template_name
        self._template_text = template_text

    def estimate_size(self) -> Optional[int]:
        count = 0
        for item in self._data:
            prompt = item.get(self._prompt_key, '')
            if not prompt:
                continue
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                return int(self.max_samples)
        return count

    def iter_items(self) -> Iterator[TaskItem]:
        output_idx = 0
        for raw_idx, item in enumerate(self._data):
            if self.max_samples is not None and output_idx >= self.max_samples:
                break
            prompt = item.get(self._prompt_key, '')
            if not prompt:
                continue
            answer = item.get(self._answer_key) if self._answer_key else None
            sample_id = str(item[self._id_key]) if self._id_key and self._id_key in item else str(raw_idx)
            meta = {key: value for key, value in item.items() if key not in {self._prompt_key, self._answer_key, self._id_key}}
            yield TaskItem(sample_idx=output_idx, sample_id=sample_id, prompt_text=None, prompt_fields={'prompt': prompt}, ground_truth=answer, meta=meta)
            output_idx += 1

    def get_prompt_template_for_item(self, item: Optional[TaskItem] = None) -> PromptTemplate:
        return PromptTemplate(name=self._template_name, template=self._template_text, variables=('prompt',))
