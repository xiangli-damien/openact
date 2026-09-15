from typing import Iterator, Optional

from openact_collect.data import HFDatasetSpec
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
            self._dataset = self.load_hf_dataset(
                HFDatasetSpec(name="allenai/ai2_arc", config="ARC-Challenge", split=self.split)
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
            question = item["question"]
            original_labels = item["choices"]["label"]
            labels = list('ABCDE'[:len(original_labels)])
            texts = item["choices"]["text"]
            choices_text = "\n".join(
                f"{label}. {text}" for label, text in zip(labels, texts)
            )
            choice_labels = "".join(labels)
            answer_key = labels[original_labels.index(item["answerKey"])]
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"arc_challenge_{idx}",
                prompt_fields={
                    "question": question,
                    "choices_text": choices_text,
                    "choice_labels": choice_labels,
                },
                ground_truth=answer_key,
                meta={
                    "question": question,
                    "choices": dict(zip(labels, texts)),
                    "answer_key": answer_key,
                    "original_answer_key": item["answerKey"],
                    "original_choice_labels": original_labels,
                    "item_id": item.get("id", str(idx)),
                },
            )
