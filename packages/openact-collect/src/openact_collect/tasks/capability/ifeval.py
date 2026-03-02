from typing import Iterator, Optional
from datasets import load_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


@TaskRegistry.register("ifeval")
class IFEvalTask(Task):
    task_name = "ifeval"
    source = "google/IFEval"
    split = "train"
    language = "en"
    default_template = "raw"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "train",
        template: str = "raw",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                "google/IFEval", split=self.split
            )

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            prompt = item["prompt"]
            instruction_ids = item.get("instruction_id_list", [])
            kwargs_list = item.get("kwargs", [])
            prompt_text = template.format_safe(prompt=prompt)
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"ifeval_{idx}",
                prompt_text=prompt_text,
                ground_truth=None,
                meta={
                    "prompt": prompt,
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs_list,
                    "key": item.get("key", idx),
                },
            )
