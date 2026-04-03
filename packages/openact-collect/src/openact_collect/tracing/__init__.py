"""Activation tracing utilities.

This module provides lightweight, dependency-free activation hooks for common
Hugging Face decoder-only Transformer architectures.

The design goal is simple:
  - capture only what we need (usually the *last token* for each forward call)
  - keep model execution code readable
  - keep storage code independent from model internals
"""

from openact_collect.tracing.activation_recorder import ActivationRecorder, ActivationTrace

__all__ = [
    "ActivationRecorder",
    "ActivationTrace",
]
