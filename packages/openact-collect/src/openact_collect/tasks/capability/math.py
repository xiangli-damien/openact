import logging
from typing import Iterator, List, Optional

from openact_collect.data import HFDatasetSpec, load_hf_dataset
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
    depth = 0
    start = None
    i = 0
    while i < len(text):
        if text[i : i + 7] == "\\boxed{":
            if depth == 0:
                start = i + 7
            depth += 1
            i += 7
        elif text[i] == "{":
            depth += 1
            i += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i]
            i += 1
        else:
            i += 1
    return None
_ELEUTHERAI_SOURCE = "EleutherAI/hendrycks_math"


def _load_eleutherai(split: str, categories: List[str]) -> list:
    """Load MATH categories from the canonical EleutherAI dataset.

    We try to load each category independently so one missing config does not
    sink the entire run.
    """

    items: list = []
    failed: List[str] = []
    for cat in categories:
        try:
            ds = load_hf_dataset(HFDatasetSpec(name=_ELEUTHERAI_SOURCE, config=cat, split=split))
            for row in ds:
                row["_category"] = cat
                items.append(row)
        except Exception as exc:  # noqa: BLE001
            failed.append(cat)
            logger.warning("MATH: failed to load category '%s' (%s). Skipping.", cat, exc)
            continue
    if failed and items:
        logger.warning("MATH: loaded %d categories, skipped %d.", len(categories) - len(failed), len(failed))
    return items


_FALLBACK_SOURCES = [
    "DigitalLearningGmbH/MATH-lighteval",
    "lighteval/MATH-Hard",
    "hendrycks/competition_math",
]


def _load_fallback(split: str) -> list:
    for hf_id in _FALLBACK_SOURCES:
        try:
            ds = load_hf_dataset(HFDatasetSpec(name=hf_id, split=split))
            return list(ds)
        except Exception:
            continue
    return []
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
        items = _load_eleutherai(self.split, self.categories)
        if not items:
            items = _load_fallback(self.split)
        if not items:
            raise RuntimeError(
                "Could not load the MATH dataset from any known source. "
                "Please check your network / HuggingFace token."
            )
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
