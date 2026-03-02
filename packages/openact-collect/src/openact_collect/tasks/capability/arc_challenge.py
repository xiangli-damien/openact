"""
ARC-Challenge data-loading task.
"""

from typing import Iterator, Optional

from datasets import load_dataset

from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


@TaskRegistry.register("arc_challenge")
class ARCChallengeTask(Task):
    task_name = "arc_challenge"
    source = "allenai/ai2_arc"
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
        super().__init__(max_samples=max_samples, split=split, template=template, **kwargs)
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                "allenai/ai2_arc", "ARC-Challenge", split=self.split
            )

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
            choice_labels = "".join(labels)
            answer_key = item["answerKey"]

            prompt_text = template.format_safe(
                question=question,
                choices_text=choices_text,
                choice_labels=choice_labels,
            )

            yield TaskItem(
                sample_idx=idx,
                sample_id=f"arc_challenge_{idx}",
                prompt_text=prompt_text,
                ground_truth=answer_key,
                meta={
                    "question": question,
                    "choices": dict(zip(labels, texts)),
                    "answer_key": answer_key,
                    "item_id": item.get("id", str(idx)),
                },
            )
