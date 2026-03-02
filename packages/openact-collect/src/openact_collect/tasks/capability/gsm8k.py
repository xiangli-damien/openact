from typing import Iterator, Optional
from datasets import load_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


@TaskRegistry.register("gsm8k")
class GSM8KTask(Task):
    task_name = "gsm8k"
    source = "openai/gsm8k"
    split = "test"
    language = "en"
    default_template = "zot"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: str = "zot",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                "openai/gsm8k", "main", split=self.split
            )

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            prompt_text = template.format_safe(question=item["question"])
            answer = item.get("answer", "")
            numeric_answer = None
            if "####" in answer:
                numeric_answer = answer.split("####")[-1].strip()
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"gsm8k_{idx}",
                prompt_text=prompt_text,
                ground_truth=numeric_answer,
                meta={
                    "question": item["question"],
                    "full_answer": answer,
                },
            )
