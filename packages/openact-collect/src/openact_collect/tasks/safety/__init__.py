from openact_collect.tasks.base import GenericTask, Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask, SafetyTaskItem
TaskRegistry._tasks.setdefault("generic", GenericTask)
__all__ = [
    "Task",
    "TaskItem",
    "GenericTask",
    "TaskRegistry",
    "SafetyTask",
    "SafetyTaskItem",
]
