from typing import Iterator, Optional
from datasets import load_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry

MGSM_LANGUAGES = [
    "bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"
]


@TaskRegistry.register("mgsm")
class MGSMTask(Task):
    task_name = "mgsm"
    source = "juletxara/mgsm"
    split = "test"
    language = "en"
    default_template = "zot"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: str = "zot",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        if language not in MGSM_LANGUAGES:
            available = ", ".join(MGSM_LANGUAGES)
            raise ValueError(
                f"Unsupported language '{language}' for MGSM. "
                f"Available: {available}"
            )
        self.language = language
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                "juletxara/mgsm", self.language, split=self.split
            )

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            question = item.get("question", "")
            if not question:
                continue
            answer_number = item.get("answer_number", None)
            numeric_answer = None
            if answer_number is not None:
                numeric_answer = str(answer_number).strip()
            answer_text = item.get("answer", "")
            prompt_text = template.format_safe(question=question)
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"mgsm_{self.language}_{idx}",
                prompt_text=prompt_text,
                ground_truth=numeric_answer,
                language=self.language,
                meta={
                    "question": question,
                    "full_answer": answer_text,
                    "answer_number": answer_number,
                    "language": self.language,
                },
            )
