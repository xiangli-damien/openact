from typing import Iterator, Optional

from openact_collect.data import HFDatasetSpec, load_hf_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry
@TaskRegistry.register("theoremqa")
class TheoremQATask(Task):
    task_name = "theoremqa"
    source = "TIGER-Lab/TheoremQA"
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
            self._dataset = load_hf_dataset(
                HFDatasetSpec(name="TIGER-Lab/TheoremQA", split=self.split)
            )

    def estimate_size(self) -> Optional[int]:
        self._load_dataset()
        if self._dataset is None:
            return None
        n = 0
        for item in self._dataset:
            if item.get("Picture", None):
                continue
            n += 1
            if self.max_samples and n >= self.max_samples:
                return int(self.max_samples)
        return n
    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        idx = 0
        for item in self._dataset:
            if self.max_samples and idx >= self.max_samples:
                break
            question = item.get("Question", item.get("question", ""))
            answer = item.get("Answer", item.get("answer", ""))
            answer_type = item.get(
                "Answer_type", item.get("answer_type", "float")
            )
            if item.get("Picture", None):
                continue
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"theoremqa_{idx}",
                prompt_fields={
                    "question": question,
                    "answer_type": answer_type,
                },
                ground_truth=str(answer),
                meta={
                    "question": question,
                    "answer_type": answer_type,
                },
            )
            idx += 1
