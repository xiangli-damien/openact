from dataclasses import dataclass, field
from typing import Any, Dict, List
from openact_collect.schema.generation import GenerationProfile
DEFAULT_SAFETY_PROFILES = {
    "greedy": GenerationProfile(name="greedy", temperature=0.0, do_sample=False, n_gen=1),
    "warm": GenerationProfile(name="warm", temperature=0.7, do_sample=True, n_gen=2),
    "hot": GenerationProfile(name="hot", temperature=1.0, do_sample=True, n_gen=2),
}
@dataclass
class SafetySpec:
    is_safety_run: bool = False
    safety_dataset: str = ""
    attack_methods: List[str] = field(default_factory=list)
    prompt_variant_types: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetySpec":
        return cls(**d)
def build_profiles_from_cli(profile_str: str) -> Dict[str, GenerationProfile]:
    profiles = {}
    for part in profile_str.split():
        if ":" not in part:
            continue
        name, params_str = part.split(":", 1)
        params: Dict[str, Any] = {}
        for kv in params_str.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in ("temperature", "top_p"):
                params[k] = float(v)
            elif k in ("top_k", "max_new_tokens", "n_gen", "seed"):
                params[k] = int(v)
            elif k == "do_sample":
                params[k] = v.lower() in ("true", "1", "yes", "on")
            else:
                params[k] = v
        profiles[name] = GenerationProfile(name=name, **params)
    return profiles
