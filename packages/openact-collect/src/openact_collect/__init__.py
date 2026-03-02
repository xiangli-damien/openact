"""
OpenAct Collect - High-performance collection engine for LLM activations.

This package provides the collection infrastructure for gathering
activation data from language models at scale.

Key features:
- Async I/O: GPU inference never waits for disk writes
- Token offset computation: enables hard alignment
- Flexible capture: hidden states, attention, MLP
- Robust resume: continue from interruptions

Example:
    >>> from openact_collect import CollectionRunner, ModelRunner
    >>> from openact_collect.tasks.capability import GSM8KTask
    >>> 
    >>> model = ModelRunner("Qwen/Qwen2-7B-Instruct")
    >>> task = GSM8KTask(max_samples=1000)
    >>> collector = CollectionRunner(model, task, output_dir="runs/gsm8k")
    >>> collector.run()
"""

__version__ = "0.1.0"

from openact_collect.engine.collector import CollectionRunner
from openact_collect.engine.model_manager import ModelRunner
from openact_collect.engine.async_writer import AsyncZarrWriter
from openact_collect.extractors.base import Extractor
from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


# Simple aliases for backward compatibility
Collector = CollectionRunner
ModelManager = ModelRunner


__all__ = [
    "__version__",
    # Primary names
    "CollectionRunner",
    "ModelRunner",
    # Backward compatibility aliases
    "Collector",
    "ModelManager",
    # Other exports
    "AsyncZarrWriter",
    "Extractor",
    "HiddenStateExtractor",
    "Task",
    "TaskItem",
    "TaskRegistry",
]
