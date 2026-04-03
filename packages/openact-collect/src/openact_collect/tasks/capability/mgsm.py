from typing import Iterator, Optional

from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry

_MGSM_TSV_BASE = "https://huggingface.co/datasets/juletxara/mgsm/resolve/main"
MGSM_LANGUAGES = [
    "bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"
]
@TaskRegistry.register("mgsm")
class MGSMTask(Task):
    task_name = "mgsm"
    source = "juletxara/mgsm"
    split = "test"
    language = "en"
    default_template = "zot"
    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: str = "zot",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        if language not in MGSM_LANGUAGES:
            available = ", ".join(MGSM_LANGUAGES)
            raise ValueError(
                f"Unsupported language '{language}' for MGSM. "
                f"Available: {available}"
            )
        self.language = language
        self._dataset = None
    def _load_dataset(self):
        if self._dataset is None:
            from datasets import load_dataset

            url = f"{_MGSM_TSV_BASE}/mgsm_{self.language}.tsv"
            self._dataset = load_dataset(
                "csv",
                data_files=url,
                delimiter="\t",
                column_names=["question", "answer"],
                split="train",
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
            question = item.get("question", "")
            if not question:
                continue
            answer_text = item.get("answer", "")
            answer_number = item.get("answer_number", None)
            if answer_number is None and answer_text is not None:
                answer_number = answer_text
            numeric_answer = None
            if answer_number is not None:
                numeric_answer = str(answer_number).strip()
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"mgsm_{self.language}_{idx}",
                prompt_fields={"question": question},
                ground_truth=numeric_answer,
                language=self.language,
                meta={
                    "question": question,
                    "full_answer": str(answer_text) if answer_text else "",
                    "answer_number": answer_number,
                    "language": self.language,
                },
            )
