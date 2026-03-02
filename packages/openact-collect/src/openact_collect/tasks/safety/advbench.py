"""
AdvBench safety data collection task.

Dataset: Zou et al. (GCG paper) harmful behavior set.
  - 520 harmful behaviors (harmful split only)

Scale estimation (greedy + warm×2 + hot×2 = 5 gens):
  - 520 × 1 variant × 5 = 2600
  - With higher n_gen: 520 × 1 × 7 = 3640

To reach 3k-5k, either increase n_gen in profiles or use all 520 behaviors.

Supported HuggingFace sources (tried in order):
  1. ``walledai/AdvBench``
  2. ``manueldeprada/AdvBench``
"""

from typing import Any, Dict, List, Optional

from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask


_HF_SOURCES = [
    ("walledai/AdvBench", None, "train"),
    ("manueldeprada/AdvBench", None, "train"),
]


@TaskRegistry.register("advbench")
class AdvBenchTask(SafetyTask):
    task_name = "advbench"
    source = "walledai/AdvBench"
    language = "en"
    default_template = "raw"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "harmful",
        template: Optional[str] = None,
        profiles: Optional[Dict[str, GenerationProfile]] = None,
        **kwargs,
    ):
        # AdvBench is harmful-only by nature; override split
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
        # force harmful in active splits
        if "harmful" not in self._active_splits:
            self._active_splits.append("harmful")

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
                "Could not load AdvBench from any known HuggingFace source. "
                f"Tried: {[s[0] for s in _HF_SOURCES]}"
            )

        behaviors: List[Dict[str, Any]] = []
        for i, item in enumerate(dataset):
            # Column names vary across sources
            goal = (
                item.get("prompt")
                or item.get("goal")
                or item.get("text")
                or ""
            )
            if not goal.strip():
                continue

            target = item.get("target", item.get("response", ""))
            behaviors.append(
                {
                    "behavior_id": f"advbench_{i:04d}",
                    "goal": goal.strip(),
                    "target": target,
                    "category": "harmful",
                    "source": "advbench",
                    "split": "harmful",
                }
            )

        if not behaviors:
            warnings.warn(
                "AdvBench loaded but contained no usable behaviors.",
                UserWarning,
            )

        return behaviors
