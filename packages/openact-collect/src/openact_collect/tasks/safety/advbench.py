from typing import Any, Dict, List, Optional
from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask
_HF_SOURCES = [
    ("S3IC/advbench", None, "train"),
    ("walledai/AdvBench", None, "train"),
    ("manueldeprada/AdvBench", None, "train"),
]
@TaskRegistry.register("advbench")
class AdvBenchTask(SafetyTask):
    task_name = "advbench"
    source = "S3IC/advbench"
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
        if "harmful" not in self._active_splits:
            self._active_splits.append("harmful")
    def load_behaviors(self) -> List[Dict[str, Any]]:
        from openact_collect.data import HFDatasetSpec, load_hf_dataset
        import warnings
        dataset = None
        for hf_id, cfg, split_name in _HF_SOURCES:
            try:
                dataset = load_hf_dataset(HFDatasetSpec(name=hf_id, config=cfg, split=split_name))
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
