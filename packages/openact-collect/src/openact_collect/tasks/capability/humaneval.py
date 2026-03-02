from typing import Iterator, Optional
from datasets import load_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


@TaskRegistry.register("humaneval")
class HumanEvalTask(Task):
    task_name = "humaneval"
    source = "openai/openai_humaneval"
    split = "test"
    language = "en"
    default_template = "instruct"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: str = "instruct",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                "openai/openai_humaneval", split=self.split
            )

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            task_id = item["task_id"]
            prompt = item["prompt"]
            canonical_solution = item.get("canonical_solution", "")
            entry_point = item.get("entry_point", "")
            test_code = item.get("test", "")
            prompt_text = template.format_safe(prompt=prompt)
            yield TaskItem(
                sample_idx=idx,
                sample_id=task_id,
                prompt_text=prompt_text,
                ground_truth=canonical_solution,
                meta={
                    "task_id": task_id,
                    "prompt": prompt,
                    "canonical_solution": canonical_solution,
                    "entry_point": entry_point,
                    "test": test_code,
                },
            )
