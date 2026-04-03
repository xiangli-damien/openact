import warnings
from typing import Dict, Type, Optional, List, Any

class TaskRegistry:
    _tasks: Dict[str, Type] = {}
    _loaded_modules: set = set()
    _all_loaded: bool = False
    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: Type):
            cls._tasks[name] = task_cls
            task_cls.task_name = name
            cls._validate_descriptor(name, task_cls)
            return task_cls
        return decorator
    @classmethod
    def _validate_descriptor(cls, name: str, task_cls: Type) -> None:
        try:
            from openact_core.tasks.descriptor import has_descriptor, get_descriptor
            if not has_descriptor(name):
                warnings.warn(
                    f"Task '{name}' registered without a matching TaskDescriptor in "
                    f"openact_core.tasks.descriptor.TASK_DESCRIPTORS. Evaluation and "
                    f"parsing may not work correctly.",
                    UserWarning,
                    stacklevel=3,
                )
                return
            desc = get_descriptor(name)
            task_source = getattr(task_cls, "source", None)
            if task_source and desc.source and task_source != desc.source:
                warnings.warn(
                    f"Task '{name}' source mismatch: class has '{task_source}', "
                    f"descriptor has '{desc.source}'.",
                    UserWarning,
                    stacklevel=3,
                )
        except ImportError:
            pass
    @classmethod
    def _ensure_loaded(cls, name: str) -> None:
        if name in cls._tasks:
            return
        name_lower = name.lower()
        if name_lower in ("jbb", "advbench", "xstest"):
            cls._load_safety_tasks()
        else:
            cls._load_capability_tasks()
    @classmethod
    def _load_capability_tasks(cls) -> None:
        if "capability" in cls._loaded_modules:
            return
        cls._loaded_modules.add("capability")
        try:
            from openact_collect.tasks import capability  # noqa: F401
            from openact_collect.tasks import prepared  # noqa: F401
        except ImportError:
            pass
    @classmethod
    def _load_safety_tasks(cls) -> None:
        if "safety" in cls._loaded_modules:
            return
        cls._loaded_modules.add("safety")
        try:
            from openact_collect.tasks.safety import jbb, advbench, xstest  # noqa: F401
        except ImportError:
            pass
    @classmethod
    def _load_all(cls) -> None:
        if cls._all_loaded:
            return
        cls._all_loaded = True
        cls._load_capability_tasks()
        cls._load_safety_tasks()
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        cls._ensure_loaded(name)
        return cls._tasks.get(name)
    @classmethod
    def create(cls, name: str, **kwargs):
        task_cls = cls.get(name)
        if task_cls is None:
            cls._load_all()
            task_cls = cls._tasks.get(name)
        if task_cls is None:
            available = ", ".join(cls.list())
            raise ValueError(f"Task '{name}' not found. Available: {available}")
        return task_cls(**kwargs)
    @classmethod
    def list(cls) -> List[str]:
        cls._load_all()
        return sorted(cls._tasks.keys())
    @classmethod
    def list_loaded(cls) -> List[str]:
        return sorted(cls._tasks.keys())
    @classmethod
    def list_with_info(cls) -> Dict[str, Dict[str, Any]]:
        cls._load_all()
        result = {}
        for name, task_cls in cls._tasks.items():
            result[name] = {
                "class": task_cls.__name__,
                "source": getattr(task_cls, "source", "unknown"),
                "language": getattr(task_cls, "language", "en"),
                "doc": task_cls.__doc__ or "",
            }
        return result
    @classmethod
    def is_loaded(cls, name: str) -> bool:
        return name in cls._tasks
    @classmethod
    def clear(cls) -> None:
        cls._tasks.clear()
        cls._loaded_modules.clear()
        cls._all_loaded = False
def _register_generic_task() -> None:
    if "generic" not in TaskRegistry._tasks:
        from openact_collect.tasks.base import GenericTask
        TaskRegistry._tasks["generic"] = GenericTask
        GenericTask.task_name = "generic"
_register_generic_task()
