from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from openact_core.tasks.templates import PromptTemplate, get_template


@dataclass
class TaskItem:
    sample_idx: int
    sample_id: str
    prompt_text: str
    ground_truth: Optional[str] = None
    language: str = "en"
    meta: Dict[str, Any] = field(default_factory=dict)

    def parquet_meta(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"language": self.language}
        for k, v in self.meta.items():
            if k not in ("sample_id", "prompt_text", "ground_truth"):
                out[k] = v
        return out


class Task(ABC):
    task_name: str = "base_task"
    task_type: str = "capability"
    source: str = "unknown"
    split: str = "test"
    language: str = "en"
    max_samples: Optional[int] = None
    default_template: str = "cot"

    def get_profiles(self) -> Dict[str, Any]:
        """Return generation profiles for multi-generation tasks. Base: empty."""
        return {}

    def get_safety_spec(self) -> Optional[Any]:
        """Return safety run spec if this is a safety task. Base: None."""
        return None

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: Optional[str] = None,
        **kwargs,
    ):
        if max_samples is not None:
            self.max_samples = max_samples
        self.split = split
        self._template_variant = template or self.default_template
        self._config = kwargs

    @abstractmethod
    def iter_items(self) -> Iterator[TaskItem]:
        ...

    def estimate_size(self) -> Optional[int]:
        return None

    def get_prompt_template(self) -> PromptTemplate:
        return get_template(self.task_name, self._template_variant)

    @property
    def name(self) -> str:
        return self.task_name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.task_name}', "
            f"split='{self.split}', template='{self._template_variant}')"
        )


class GenericTask(Task):
    task_name = "generic"
    source = "custom"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        prompt_key: str = "prompt",
        answer_key: Optional[str] = "answer",
        id_key: Optional[str] = None,
        template_name: str = "generic",
        template_text: str = "{prompt}",
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(max_samples=max_samples, **kwargs)
        self._data = data
        self._prompt_key = prompt_key
        self._answer_key = answer_key
        self._id_key = id_key
        self._template_name = template_name
        self._template_text = template_text

    def estimate_size(self) -> Optional[int]:
        n = len(self._data)
        if self.max_samples:
            n = min(n, self.max_samples)
        return n

    def iter_items(self) -> Iterator[TaskItem]:
        for idx, item in enumerate(self._data):
            if self.max_samples and idx >= self.max_samples:
                break
            prompt = item.get(self._prompt_key, "")
            if not prompt:
                continue
            answer = None
            if self._answer_key:
                answer = item.get(self._answer_key)
            sample_id = str(idx)
            if self._id_key and self._id_key in item:
                sample_id = str(item[self._id_key])
            meta = {
                k: v
                for k, v in item.items()
                if k not in {self._prompt_key, self._answer_key, self._id_key}
            }
            yield TaskItem(
                sample_idx=idx,
                sample_id=sample_id,
                prompt_text=self._template_text.format(prompt=prompt),
                ground_truth=answer,
                meta=meta,
            )

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            name=self._template_name,
            template=self._template_text,
            variables=("prompt",),
        )