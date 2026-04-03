from typing import Iterator, Optional

from openact_collect.data import HFDatasetSpec, load_hf_dataset
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
            self._dataset = load_hf_dataset(HFDatasetSpec(name="google/IFEval", split=self.split))

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
            prompt = item["prompt"]
            instruction_ids = item.get("instruction_id_list", [])
            kwargs_list = item.get("kwargs", [])
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"ifeval_{idx}",
                prompt_fields={"prompt": prompt},
                ground_truth=None,
                meta={
                    "prompt": prompt,
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs_list,
                    "key": item.get("key", idx),
                },
            )
