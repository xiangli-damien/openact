"""
OpenAct Collect Schema - Configuration dataclasses.

This module provides configuration specifications for:
- CaptureSpec: What to capture (hidden states, layers, etc.)
- StorageSpec: How to store (compression, chunking)
- GenerationSpec: Generation parameters (temperature, top_p, etc.)
- GenerationProfile: Named generation profiles for multi-generation tasks
- SafetySpec: Safety-specific run configuration
"""

from openact_collect.schema.capture import CaptureSpec
from openact_collect.schema.storage import StorageSpec
# GenerationSpec and GenerationProfile are both defined in generation.py
from openact_collect.schema.generation import GenerationSpec, GenerationProfile
# SafetySpec and profile helpers are in safety.py (which imports GenerationProfile from generation.py)
from openact_collect.schema.safety import SafetySpec, DEFAULT_SAFETY_PROFILES, build_profiles_from_cli

__all__ = [
    'CaptureSpec',
    'StorageSpec',
    'GenerationSpec',
    'GenerationProfile',
    'SafetySpec',
    'DEFAULT_SAFETY_PROFILES',
    'build_profiles_from_cli',
]
