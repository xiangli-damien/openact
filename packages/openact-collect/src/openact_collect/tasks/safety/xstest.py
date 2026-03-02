"""
XSTest safety data collection task.

Dataset: Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated
Safety Behaviours in Large Language Models"

  - ~250 *safe* prompts that well-aligned models should answer (split="benign")
  - ~200 *unsafe* prompts that models should refuse (split="harmful")

The key research value is measuring **over-refusal**: does the model refuse
benign prompts that merely *sound* dangerous?

Scale estimation (greedy + warm×2 + hot×2 = 5 gens):
  - 450 × 5 = 2250 total
  - With higher n_gen: 450 × 7 = 3150

Supported HuggingFace sources:
  1. ``nreimers/XSTest``
  2. ``PaulR/xstest``
"""

from typing import Any, Dict, List, Optional

from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask


_HF_SOURCES = [
    ("nreimers/XSTest", None, "test"),
    ("PaulR/xstest", None, "test"),
]


@TaskRegistry.register("xstest")
class XSTestTask(SafetyTask):
    task_name = "xstest"
    source = "nreimers/XSTest"
    language = "en"
    default_template = "raw"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "all",
        template: Optional[str] = None,
        profiles: Optional[Dict[str, GenerationProfile]] = None,
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples,
            split=split,
            template=template,
            profiles=profiles,
            include_goal=True,
            include_artifacts=False,
            artifact_loader=None,
            **kwargs,
        )

    def load_behaviors(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset
        import warnings

        dataset = None
        for hf_id, cfg, split_name in _HF_SOURCES:
            try:
                if cfg:
                    dataset = load_dataset(hf_id, cfg, split=split_name)
                else:
                    dataset = load_dataset(hf_id, split=split_name)
                self.source = hf_id
                break
            except Exception:
                continue

        if dataset is None:
            raise RuntimeError(
                "Could not load XSTest from any known HuggingFace source. "
                f"Tried: {[s[0] for s in _HF_SOURCES]}"
            )

        behaviors: List[Dict[str, Any]] = []
        for i, item in enumerate(dataset):
            prompt = (
                item.get("prompt")
                or item.get("question")
                or item.get("text")
                or ""
            )
            if not prompt.strip():
                continue

            # Determine split from label
            label = (
                item.get("label", "")
                or item.get("type", "")
                or item.get("safe", "")
            )
            label_str = str(label).lower().strip()

            if label_str in ("safe", "1", "true", "benign"):
                b_split = "benign"
            elif label_str in ("unsafe", "0", "false", "harmful"):
                b_split = "harmful"
            else:
                # Some versions use a "type" column
                xstest_type = item.get("type", "").lower()
                if "safe" in xstest_type or "contrast" in xstest_type:
                    b_split = "benign"
                else:
                    b_split = "harmful"

            category = item.get("type", item.get("category", ""))
            note = item.get("note", item.get("focus", ""))

            behaviors.append(
                {
                    "behavior_id": f"xstest_{i:04d}",
                    "goal": prompt.strip(),
                    "category": str(category),
                    "source": "xstest",
                    "split": b_split,
                    "note": str(note),
                }
            )

        if not behaviors:
            warnings.warn(
                "XSTest loaded but contained no usable prompts.", UserWarning
            )

        return behaviors
