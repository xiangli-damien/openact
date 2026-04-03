from typing import Any, Dict, List, Optional
from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask
_HF_SOURCES = [
    ("Paul/XSTest", None, "train"),
    ("AlignmentResearch/XSTest", None, "validation"),
]
@TaskRegistry.register("xstest")
class XSTestTask(SafetyTask):
    task_name = "xstest"
    source = "Paul/XSTest"
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
                content = item.get("content")
                if isinstance(content, (list, tuple)) and content:
                    prompt = str(content[0])
                elif isinstance(content, str):
                    prompt = content
            if not str(prompt).strip():
                continue
            prompt = str(prompt).strip()
            label = (
                item.get("label")
                or item.get("gen_target")
                or item.get("proxy_gen_target")
                or item.get("safe", "")
            )
            label_str = str(label).lower().strip()
            if label_str in ("safe", "1", "true", "benign"):
                b_split = "benign"
            elif label_str in ("unsafe", "0", "false", "harmful"):
                b_split = "harmful"
            else:
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
                    "goal": prompt,
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
