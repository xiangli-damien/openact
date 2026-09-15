from typing import Any, Dict, List, Optional
from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask
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
        from openact_collect.data import HFDatasetSpec
        import warnings
        dataset = self.load_hf_dataset(HFDatasetSpec(name=self.source, split='train'))
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
            label = item.get('label')
            label_str = str(label).lower().strip()
            if label_str in ('safe', '1', 'true', 'benign'):
                b_split = 'benign'
            elif label_str in ('unsafe', '0', 'false', 'harmful'):
                b_split = 'harmful'
            else:
                raise ValueError(f'Unrecognized XSTest safety label: {label!r}')
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
