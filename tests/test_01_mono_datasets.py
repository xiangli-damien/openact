"""01 — Monolingual dataset download & field inspection.

Downloads each dataset (max_samples=5), prints every field of first 2 items.
Verifies ground_truth format is correct for the task type.

Run:
    python -m pytest tests/test_01_mono_datasets.py -v -s --tb=short
    python -m pytest tests/test_01_mono_datasets.py -v -s -k gsm8k   # single task
"""
import pytest
from conftest import MONO_TASKS


def _trunc(text, n=200):
    s = str(text) if text is not None else "None"
    return s[:n] + ("…" if len(s) > n else "")


def _print_item(item, idx):
    print(f"\n  ─── item[{idx}] ───")
    print(f"    sample_idx    = {item.sample_idx}")
    print(f"    sample_id     = {item.sample_id}")
    print(f"    ground_truth  = {_trunc(item.ground_truth, 100)}")
    print(f"    language      = {item.language}")
    if item.prompt_text is not None:
        print(f"    prompt_text   = {_trunc(item.prompt_text, 150)}")
    if item.prompt_fields:
        for k, v in item.prompt_fields.items():
            print(f"    fields[{k}] = {_trunc(v, 150)}")
    if item.meta:
        for k, v in list(item.meta.items())[:8]:
            print(f"    meta[{k}] = {_trunc(v, 100)}")


@pytest.mark.parametrize("task_name", MONO_TASKS)
def test_dataset_download_and_fields(task_name):
    from openact_collect.tasks.registry import TaskRegistry

    task = TaskRegistry.create(task_name, max_samples=5)
    items = list(task.iter_items())

    print(f"\n{'='*72}")
    print(f"  TASK: {task_name}")
    print(f"  source={task.source}  split={task.split}  template={task._template_variant}")
    print(f"  items returned: {len(items)}")
    print(f"{'='*72}")

    assert len(items) > 0, f"No items for {task_name}"
    assert len(items) <= 5

    for i, item in enumerate(items[:2]):
        _print_item(item, i)

    # ── Ground truth format checks ───────────────────────────────────
    for item in items:
        if task_name in ("gsm8k",):
            assert item.ground_truth is not None, "GSM8K must have GT"
            try:
                float(str(item.ground_truth).replace(",", ""))
            except ValueError:
                pytest.fail(f"GSM8K GT not numeric: {item.ground_truth!r}")

        elif task_name in ("mmlu", "arc_challenge", "commonsenseqa"):
            assert item.ground_truth is not None
            assert str(item.ground_truth).strip() in "ABCDE", (
                f"{task_name} GT not a letter: {item.ground_truth!r}"
            )

        elif task_name == "math":
            assert item.ground_truth is not None, "MATH must have GT"

        elif task_name == "theoremqa":
            assert item.ground_truth is not None, "TheoremQA must have GT"

        elif task_name == "truthfulqa":
            assert item.ground_truth is not None

        elif task_name in ("humaneval", "ifeval"):
            pass  # GT may be None, that's ok

    print(f"\n  ✓ Ground truth format verified for all {len(items)} items")


def test_task_registry_completeness():
    """All expected tasks are registered."""
    from openact_collect.tasks.registry import TaskRegistry

    all_tasks = TaskRegistry.list()
    print(f"\n  Registered tasks ({len(all_tasks)}): {all_tasks}")
    for t in MONO_TASKS:
        assert t in all_tasks, f"Task {t!r} not registered"
    print(f"  ✓ All {len(MONO_TASKS)} mono tasks registered")