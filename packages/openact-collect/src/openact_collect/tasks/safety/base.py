import hashlib
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from openact_collect.schema import (
    DEFAULT_SAFETY_PROFILES,
    GenerationProfile,
    SafetySpec,
)
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.safety.artifacts import ArtifactLoader
from openact_core.tasks.templates import PromptTemplate


@dataclass
class SafetyTaskItem(TaskItem):
    behavior_id: str = ""
    split: str = ""
    category: str = ""
    prompt_variant: str = ""
    attack_method: str = ""
    profile: str = "default"
    rep_idx: int = 0

    def parquet_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "language": self.language,
            "behavior_id": self.behavior_id,
            "split": self.split,
            "category": self.category,
            "prompt_variant": self.prompt_variant,
            "attack_method": self.attack_method,
            "profile": self.profile,
            "rep_idx": self.rep_idx,
        }
        if "seed" in self.meta:
            meta["seed"] = self.meta["seed"]
        return meta


class SafetyTask(Task):
    task_type: str = "safety"

    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "all",
        template: Optional[str] = None,
        profiles: Optional[Dict[str, GenerationProfile]] = None,
        include_goal: bool = True,
        include_artifacts: bool = True,
        artifact_loader: Optional[ArtifactLoader] = None,
        **kwargs,
    ):
        super().__init__(max_samples=max_samples, split=split, template=template, **kwargs)
        self.profiles = profiles or dict(DEFAULT_SAFETY_PROFILES)
        self.include_goal = include_goal
        self.include_artifacts = include_artifacts
        self.artifact_loader = artifact_loader
        self._active_splits = self._parse_splits(split)

    def get_profiles(self) -> Dict[str, GenerationProfile]:
        return self.profiles

    def get_safety_spec(self) -> Optional[SafetySpec]:
        methods: List[str] = []
        if self.artifact_loader is not None:
            methods = self.artifact_loader.list_methods()
        variant_types = ["goal"]
        if methods:
            variant_types.extend(f"artifact_{m}" for m in methods)
        return SafetySpec(
            is_safety_run=True,
            safety_dataset=self.task_name,
            attack_methods=methods,
            prompt_variant_types=variant_types,
            splits=list(self._active_splits),
            profiles={name: prof.to_dict() for name, prof in self.profiles.items()},
        )

    @staticmethod
    def _parse_splits(split: str) -> List[str]:
        if split in ("all", "*"):
            return ["harmful", "benign"]
        return [s.strip() for s in split.split(",")]

    @abstractmethod
    def load_behaviors(self) -> List[Dict[str, Any]]:
        ...

    def build_prompt_variants(
        self, behavior: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        variants: List[Tuple[str, str, str]] = []
        if self.include_goal:
            variants.append((behavior["goal"], "goal", "none"))
        if self.include_artifacts and self.artifact_loader:
            for art in self.artifact_loader.get(behavior["behavior_id"]):
                variants.append((art.prompt, f"artifact_{art.method}", art.method))
        if not variants:
            variants.append((behavior["goal"], "goal", "none"))
        return variants

    @staticmethod
    def _derive_seed(
        behavior_id: str,
        variant: str,
        profile_name: str,
        rep: int,
        base_seed: int = 42,
    ) -> int:
        key = f"{behavior_id}:{variant}:{profile_name}:{rep}:{base_seed}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16) % 2**31

    def estimate_size(self) -> Optional[int]:
        """Return None to force full plan materialization, ensuring sample_idx
        consistency between Zarr allocation and iter_items yield order.
        """
        return None

    def iter_items(self) -> Iterator[SafetyTaskItem]:
        behaviors = self.load_behaviors()
        idx = 0
        for behavior in behaviors:
            b_split = behavior.get("split", "harmful")
            if b_split not in self._active_splits:
                continue
            variants = self.build_prompt_variants(behavior)
            for prompt_text, variant_name, attack_method in variants:
                for profile_name, profile in self.profiles.items():
                    n_gen = profile.n_gen
                    for rep in range(n_gen):
                        if self.max_samples and idx >= self.max_samples:
                            return
                        seed = self._derive_seed(
                            behavior["behavior_id"],
                            variant_name,
                            profile_name,
                            rep,
                            profile.seed,
                        )
                        item = SafetyTaskItem(
                            sample_idx=idx,
                            sample_id=(
                                f"{self.task_name}_{behavior['behavior_id']}_"
                                f"{variant_name}_{profile_name}_r{rep}"
                            ),
                            prompt_text=prompt_text,
                            ground_truth=b_split,
                            language=self.language,
                            behavior_id=behavior["behavior_id"],
                            split=b_split,
                            category=behavior.get("category", ""),
                            prompt_variant=variant_name,
                            attack_method=attack_method,
                            profile=profile_name,
                            rep_idx=rep,
                            meta={
                                "source": behavior.get("source", ""),
                                "seed": seed,
                            },
                        )
                        yield item
                        idx += 1

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            name=f"{self.task_name}_raw",
            template="{prompt}",
            variables=("prompt",),
            description="Raw prompt pass-through for safety tasks.",
        )