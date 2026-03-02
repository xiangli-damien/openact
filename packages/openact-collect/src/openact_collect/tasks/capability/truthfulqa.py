from typing import Iterator, Optional, List
from datasets import load_dataset
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
            self._dataset = load_dataset(
                "truthful_qa", "generation", split=self.split
            )

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()
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
            prompt_text = template.format_safe(question=question)
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"truthfulqa_{idx}",
                prompt_text=prompt_text,
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
