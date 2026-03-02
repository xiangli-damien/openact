"""
openact_collect.tasks — Data-loading tasks for activation collection.

Each task loads data from an external source and yields ``TaskItem`` objects.
Prompt templates and answer parsers are in ``openact_core.tasks``.
"""

from openact_collect.tasks.base import Task, TaskItem, GenericTask
from openact_collect.tasks.registry import TaskRegistry

__all__ = ["Task", "TaskItem", "GenericTask", "TaskRegistry"]
