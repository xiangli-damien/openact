"""
Base matcher interface.
"""

from abc import ABC, abstractmethod


class Matcher(ABC):
    """Base class for answer matching strategies."""

    @abstractmethod
    def match(self, extracted: str, ground_truth: str) -> bool:
        """
        Check if extracted answer matches ground truth.

        Args:
            extracted: Normalized extracted answer string
            ground_truth: Normalized ground truth string

        Returns:
            True if answers match, False otherwise
        """
        ...
