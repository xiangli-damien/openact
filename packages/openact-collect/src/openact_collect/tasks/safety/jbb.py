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
        include_artifacts: bool = False,
        artifact_loader: Optional[ArtifactLoader] = None,
        artifact_methods: Optional[List[str]] = None,
        **kwargs,
    ):
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
        from openact_collect.data import HFDatasetSpec
        behaviors: List[Dict[str, Any]] = []
        for split_name in self._active_splits:
            ds_split = self.load_hf_dataset(
                HFDatasetSpec(name="JailbreakBench/JBB-Behaviors", config="behaviors", split=split_name)
            )
            for item in ds_split:
                behavior_id = (
                    item.get("BehaviorID")
                    or item.get("behavior_id")
                    or item.get("Behavior", "")
                )
                goal = (
                    item.get("Goal")
                    or item.get("goal")
                    or item.get("prompt", "")
                )
                behaviors.append(
                    {
                        "behavior_id": behavior_id,
                        "goal": goal,
                        "target": item.get("Target", item.get("target", "")),
                        "category": item.get("Category", item.get("category", "")),
                        "source": item.get("Source", item.get("source", "OriginalJBB")),
                        "split": split_name,
                    }
                )
        return behaviors
