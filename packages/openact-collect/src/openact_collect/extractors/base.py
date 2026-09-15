from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from openact_collect.extractors.hidden_state_data import HiddenStateData
from openact_collect.schema import CaptureSpec
if TYPE_CHECKING:
    from openact_collect.engine.model_manager import GenerationResult
class Extractor(ABC):
    def __init__(self, capture_spec: CaptureSpec):
        self.capture_spec = capture_spec
        self._model = None
        self._hooks: List[Any] = []
    @abstractmethod
    def setup(self, model: Any, model_manager: Any) -> None:
        pass
    @abstractmethod
    def extract(self, gen_result: "GenerationResult") -> HiddenStateData:
        pass
    @abstractmethod
    def cleanup(self) -> None:
        pass
    def get_layer_indices(self, n_layers: int) -> List[int]:
        if self.capture_spec.hidden_states_layers is not None:
            return self.capture_spec.hidden_states_layers
        return list(range(n_layers))
    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
class ExtractorRegistry:
    _extractors: Dict[str, type] = {}
    @classmethod
    def register(cls, name: str):
        def decorator(extractor_cls):
            cls._extractors[name] = extractor_cls
            return extractor_cls
        return decorator
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._extractors.get(name)
    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._extractors.keys())
