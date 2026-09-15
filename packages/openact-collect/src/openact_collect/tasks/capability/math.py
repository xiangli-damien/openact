import logging
from typing import Iterator, List, Optional

from openact_collect.data import HFDatasetSpec
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


logger = logging.getLogger(__name__)
CATEGORIES = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
LEVELS = [1, 2, 3, 4, 5]
def _extract_boxed_from_solution(text: str) -> Optional[str]:
    from openact_core.tasks.parsers.base import extract_boxed
    return extract_boxed(text)


_ELEUTHERAI_SOURCE = "EleutherAI/hendrycks_math"


@TaskRegistry.register("math")
class MATHTask(Task):
    task_name = "math"
    source = _ELEUTHERAI_SOURCE
    split = "test"
    language = "en"
    default_template = "zot"
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: str = "zot",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        self.categories = categories or CATEGORIES
        self.levels = levels or LEVELS
        self._raw_items: Optional[list] = None
    def _load_dataset(self):
        if self._raw_items is not None:
            return
        items = []
        for category in self.categories:
            # A missing category is a failed benchmark load, not permission to
            # substitute MATH-Hard or silently evaluate a partial dataset.
            dataset = self.load_hf_dataset(HFDatasetSpec(name=self.source, config=category, split=self.split))
            items.extend(dict(row, _category=category) for row in dataset)
        self._raw_items = items

    def estimate_size(self) -> Optional[int]:
        self._load_dataset()
        assert self._raw_items is not None
        n = 0
        for item in self._raw_items:
            category = (item.get("_category") or item.get("type", "")).lower()
            if category and category not in self.categories:
                continue
            level_str = item.get("level", "Level 1")
            try:
                level = int(str(level_str).replace("Level ", ""))
            except (ValueError, AttributeError):
                level = 1
            if level not in self.levels:
                continue
            if not item.get("problem", ""):
                continue
            n += 1
            if self.max_samples and n >= self.max_samples:
                return int(self.max_samples)
        return n
    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        idx = 0
        for item in self._raw_items:
            if self.max_samples and idx >= self.max_samples:
                break
            category = (
                item.get("_category") or item.get("type", "")
            ).lower()
            if category and category not in self.categories:
                continue
            level_str = item.get("level", "Level 1")
            try:
                level = int(str(level_str).replace("Level ", ""))
            except (ValueError, AttributeError):
                level = 1
            if level not in self.levels:
                continue
            problem = item.get("problem", "")
            if not problem:
                continue
            solution = item.get("solution", "")
            answer = _extract_boxed_from_solution(solution)
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"math_{idx}",
                prompt_fields={"problem": problem},
                ground_truth=answer,
                meta={
                    "problem": problem,
                    "solution": solution,
                    "category": category,
                    "level": level,
                },
            )
            idx += 1
