import json
from pathlib import Path
from typing import Any, Dict, List, Optional
class Artifact:
    __slots__ = ("behavior_id", "method", "prompt", "meta")
    def __init__(
        self,
        behavior_id: str,
        method: str,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.behavior_id = behavior_id
        self.method = method
        self.prompt = prompt
        self.meta = meta or {}
    def __repr__(self) -> str:
        return (
            f"Artifact(behavior={self.behavior_id!r}, "
            f"method={self.method!r}, len={len(self.prompt)})"
        )
class ArtifactLoader:
    def get(self, behavior_id: str) -> List[Artifact]:
        raise NotImplementedError
    def list_methods(self) -> List[str]:
        raise NotImplementedError
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
class DirectoryArtifactLoader(ArtifactLoader):
    def __init__(
        self,
        artifacts_dir: str,
        methods: Optional[List[str]] = None,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self._filter_methods = set(m.upper() for m in methods) if methods else None
        self._index: Dict[str, List[Artifact]] = {}
        self._methods: List[str] = []
        self._loaded = False
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.artifacts_dir.exists():
            return
        for jsonl in self.artifacts_dir.glob("*.jsonl"):
            method = jsonl.stem.upper()
            if self._filter_methods and method not in self._filter_methods:
                continue
            self._methods.append(method)
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    bid = obj.get("behavior_id", obj.get("BehaviorID", ""))
                    prompt = obj.get("prompt", obj.get("jailbreak_prompt", ""))
                    if bid and prompt:
                        art = Artifact(behavior_id=bid, method=method, prompt=prompt)
                        self._index.setdefault(bid, []).append(art)
        for jf in self.artifacts_dir.glob("*.json"):
            method = jf.stem.upper()
            if self._filter_methods and method not in self._filter_methods:
                continue
            if method in self._methods:
                continue
            self._methods.append(method)
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("artifacts", [])
            for obj in items:
                bid = obj.get("behavior_id", obj.get("BehaviorID", ""))
                prompt = obj.get("prompt", obj.get("jailbreak_prompt", ""))
                if bid and prompt:
                    art = Artifact(behavior_id=bid, method=method, prompt=prompt)
                    self._index.setdefault(bid, []).append(art)
        for sub in self.artifacts_dir.iterdir():
            if not sub.is_dir():
                continue
            method = sub.name.upper()
            if self._filter_methods and method not in self._filter_methods:
                continue
            if method in self._methods:
                continue
            self._methods.append(method)
            for jf in sub.glob("*.json"):
                bid = jf.stem
                with open(jf, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                prompt = obj.get("prompt", obj.get("jailbreak_prompt", ""))
                if prompt:
                    art = Artifact(behavior_id=bid, method=method, prompt=prompt)
                    self._index.setdefault(bid, []).append(art)
    def get(self, behavior_id: str) -> List[Artifact]:
        self._ensure_loaded()
        return self._index.get(behavior_id, [])
    def list_methods(self) -> List[str]:
        self._ensure_loaded()
        return list(self._methods)
    def __repr__(self) -> str:
        n = sum(len(v) for v in self._index.values()) if self._loaded else "?"
        return f"DirectoryArtifactLoader(dir={self.artifacts_dir}, n={n})"
class JBBArtifactLoader(ArtifactLoader):
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ):
        self._filter_methods = methods
        self._model_name = model_name
        self._index: Dict[str, List[Artifact]] = {}
        self._methods: List[str] = []
        self._loaded = False
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from openact_collect.data import HFDatasetSpec, load_hf_dataset
            for split_name in ("train", "test", "validation"):
                try:
                    ds_split = load_hf_dataset(
                        HFDatasetSpec(name="JailbreakBench/JBB-Behaviors", config="artifacts", split=split_name)
                    )
                except Exception:
                    continue
                for item in ds_split:
                    bid = item.get("BehaviorID", item.get("behavior_id", ""))
                    for key in item:
                        if key in ("BehaviorID", "behavior_id", "Goal", "Target"):
                            continue
                        method = key.upper()
                        if self._filter_methods and method not in [
                            m.upper() for m in self._filter_methods
                        ]:
                            continue
                        prompt = item[key]
                        if not prompt or not isinstance(prompt, str):
                            continue
                        if method not in self._methods:
                            self._methods.append(method)
                        art = Artifact(
                            behavior_id=bid, method=method, prompt=prompt
                        )
                        self._index.setdefault(bid, []).append(art)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to load JBB artifacts from HuggingFace: {e}. "
                "Continuing without artifacts.",
                UserWarning,
            )
    def get(self, behavior_id: str) -> List[Artifact]:
        self._ensure_loaded()
        return self._index.get(behavior_id, [])
    def list_methods(self) -> List[str]:
        self._ensure_loaded()
        return list(self._methods)
class DictArtifactLoader(ArtifactLoader):
    def __init__(self, mapping: Dict[str, List[tuple]]):
        self._index: Dict[str, List[Artifact]] = {}
        methods_set: set = set()
        for bid, entries in mapping.items():
            for method, prompt in entries:
                methods_set.add(method)
                self._index.setdefault(bid, []).append(
                    Artifact(behavior_id=bid, method=method, prompt=prompt)
                )
        self._methods = sorted(methods_set)
    def get(self, behavior_id: str) -> List[Artifact]:
        return self._index.get(behavior_id, [])
    def list_methods(self) -> List[str]:
        return list(self._methods)
