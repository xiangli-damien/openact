from typing import Iterator, Optional, List
from datasets import load_dataset
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry

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
    all_items: list = []
    failed_categories = []
    for cat in categories:
        try:
            ds = load_dataset(_ELEUTHERAI_SOURCE, cat, split=split)
            for item in ds:
                item["_category"] = cat
                all_items.append(item)
        except Exception as e:
            failed_categories.append(cat)
            import warnings
            warnings.warn(
                f"Failed to load MATH category '{cat}': {e}. "
                f"This category will be skipped.",
                UserWarning
            )
    if failed_categories:
        loaded = set(categories) - set(failed_categories)
        import warnings
        warnings.warn(
            f"MATH dataset: loaded {len(loaded)}/{len(categories)} categories. "
            f"Failed: {failed_categories}",
            UserWarning
        )
    return all_items


_FALLBACK_SOURCES = [
    ("DigitalLearningGmbH/MATH-lighteval", None),
    ("lighteval/MATH-Hard", None),
    ("hendrycks/competition_math", None),
]


def _load_fallback(split: str) -> list:
    for hf_id, cfg in _FALLBACK_SOURCES:
        try:
            if cfg:
                ds = load_dataset(hf_id, cfg, split=split)
            else:
                ds = load_dataset(hf_id, split=split)
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

    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        template = self.get_prompt_template()
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
            prompt_text = template.format_safe(problem=problem)
            solution = item.get("solution", "")
            answer = _extract_boxed_from_solution(solution)
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"math_{idx}",
                prompt_text=prompt_text,
                ground_truth=answer,
                meta={
                    "problem": problem,
                    "solution": solution,
                    "category": category,
                    "level": level,
                },
            )
            idx += 1
