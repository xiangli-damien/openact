from typing import Iterator, List, Optional

from openact_collect.data import HFDatasetSpec, load_hf_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry
@TaskRegistry.register("truthfulqa")
class TruthfulQATask(Task):
    task_name = "truthfulqa"
    source = "truthful_qa"
    split = "validation"
    language = "en"
    default_template = "zot"
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        split: str = "validation",
        template: str = "zot",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        self.categories = categories
        self._dataset = None
    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_hf_dataset(
                HFDatasetSpec(name="truthful_qa", config="generation", split=self.split)
            )

    def estimate_size(self) -> Optional[int]:
        self._load_dataset()
        if self._dataset is None:
            return None
        if not self.categories:
            try:
                n = len(self._dataset)
            except Exception:
                return None
            return min(n, self.max_samples) if self.max_samples else n

        # Category-filtered count.
        n = 0
        for item in self._dataset:
            if item.get("category", "") not in self.categories:
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
            category = item.get("category", "")
            if self.categories and category not in self.categories:
                continue
            question = item["question"]
            best_answer = item.get("best_answer", "")
            correct_answers = item.get("correct_answers", [])
            incorrect_answers = item.get("incorrect_answers", [])
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"truthfulqa_{idx}",
                prompt_fields={"question": question},
                ground_truth=best_answer,
                meta={
                    "question": question,
                    "best_answer": best_answer,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "category": category,
                    "source": item.get("source", ""),
                },
            )
            idx += 1
