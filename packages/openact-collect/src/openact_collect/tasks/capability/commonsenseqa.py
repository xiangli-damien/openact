"""
CommonsenseQA data-loading task.
"""

from typing import Iterator, Optional

from datasets import load_dataset

from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


@TaskRegistry.register("commonsenseqa")
class CommonsenseQATask(Task):
    task_name = "commonsenseqa"
    source = "tau/commonsense_qa"
    split = "validation"
    language = "en"
    default_template = "zot"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "validation",
        template: str = "zot",
        **kwargs,
    ):
        super().__init__(max_samples=max_samples, split=split, template=template, **kwargs)
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset("tau/commonsense_qa", split=self.split)

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()

        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break

            question = item["question"]
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            choices_text = "\n".join(
                f"{label}. {text}" for label, text in zip(labels, texts)
            )
            answer_key = item.get("answerKey", "")

            if self._template_variant == "zot":
                question_full = f"{question}\n\n{choices_text}"
                prompt_text = template.format_safe(question=question_full)
            else:
                prompt_text = template.format_safe(
                    question=question, choices_text=choices_text
                )

            yield TaskItem(
                sample_idx=idx,
                sample_id=f"commonsenseqa_{idx}",
                prompt_text=prompt_text,
                ground_truth=answer_key,
                meta={
                    "question": question,
                    "choices": dict(zip(labels, texts)),
                    "answer_key": answer_key,
                    "concept": item.get("question_concept", ""),
                },
            )
