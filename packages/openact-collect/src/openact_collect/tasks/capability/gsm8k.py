from typing import Iterator, Optional

from openact_collect.data import HFDatasetSpec
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
            self._dataset = self.load_hf_dataset(
                HFDatasetSpec(name="openai/gsm8k", config="main", split=self.split)
            )

    def estimate_size(self) -> Optional[int]:
        self._load_dataset()
        try:
            n = len(self._dataset)
        except Exception:
            return None
        return min(n, self.max_samples) if self.max_samples else n
    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            answer = item.get("answer", "")
            numeric_answer = None
            if "####" in answer:
                numeric_answer = answer.split("####")[-1].strip()
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"gsm8k_{idx}",
                prompt_fields={"question": item["question"]},
                ground_truth=numeric_answer,
                meta={
                    "question": item["question"],
                    "full_answer": answer,
                },
            )
