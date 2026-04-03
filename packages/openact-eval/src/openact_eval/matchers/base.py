from abc import ABC, abstractmethod
class Matcher(ABC):
    @abstractmethod
    def match(self, extracted: str, ground_truth: str) -> bool:
        ...
