"""
JailbreakBench (JBB) safety data collection task.

Dataset: ``JailbreakBench/JBB-Behaviors``
  - 100 harmful behaviors  (split="harmful")
  - 100 benign behaviors   (split="benign")

Scale estimation (all profiles = greedy + warm×2 + hot×2 = 5 gens):
  - Goal only:       100 harmful × 5 = 500
  - + 2 artifacts:   100 × 3 variants × 5 = 1500
  - + benign:        + 100 × 1 variant × 5 = 500
  - Total ≈ 2000 (goal-only) to 5400 (with artifacts + benign)

To reach 3k-5k *harmful* samples, use artifacts and/or increase n_gen.
"""

from typing import Any, Dict, List, Optional

from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask
from openact_collect.tasks.safety.artifacts import (
    ArtifactLoader,
    JBBArtifactLoader,
)


@TaskRegistry.register("jbb")
class JBBTask(SafetyTask):
    task_name = "jbb"
    source = "JailbreakBench/JBB-Behaviors"
    language = "en"
    default_template = "raw"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "all",
        template: Optional[str] = None,
        profiles: Optional[Dict[str, GenerationProfile]] = None,
        include_goal: bool = True,
        include_artifacts: bool = True,
        artifact_loader: Optional[ArtifactLoader] = None,
        artifact_methods: Optional[List[str]] = None,
        **kwargs,
    ):
        # Auto-create JBB artifact loader if not provided
        if include_artifacts and artifact_loader is None:
            artifact_loader = JBBArtifactLoader(methods=artifact_methods)

        super().__init__(
            max_samples=max_samples,
            split=split,
            template=template,
            profiles=profiles,
            include_goal=include_goal,
            include_artifacts=include_artifacts,
            artifact_loader=artifact_loader,
            **kwargs,
        )

    def load_behaviors(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        behaviors: List[Dict[str, Any]] = []

        for split_name in self._active_splits:
            if split_name not in ds:
                continue
            for item in ds[split_name]:
                behaviors.append(
                    {
                        "behavior_id": item["BehaviorID"],
                        "goal": item["Goal"],
                        "target": item.get("Target", ""),
                        "category": item.get("Category", ""),
                        "source": item.get("Source", "OriginalJBB"),
                        "split": split_name,
                    }
                )
        return behaviors
