"""
OpenAct Collect Engine - Core collection infrastructure.

This module provides the main engine components for data collection:
- CollectionRunner: Orchestrates the complete collection pipeline
- ModelRunner: Handles model loading and inference
- AsyncZarrWriter: Async I/O for tensor storage
- OffsetCalculator: Token-character alignment
- OnlineMetricsProcessor: Per-token metrics computation
"""

from openact_collect.engine.collector import CollectionRunner
from openact_collect.engine.model_manager import ModelRunner
from openact_collect.engine.async_writer import AsyncZarrWriter
from openact_collect.engine.offset_calculator import OffsetCalculator
from openact_collect.engine.online_metrics import OnlineMetricsProcessor

__all__ = [
    'CollectionRunner',
    'ModelRunner',
    'AsyncZarrWriter',
    'OffsetCalculator',
    'OnlineMetricsProcessor',
]