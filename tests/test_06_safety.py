"""06 — Safety datasets: JBB, AdvBench, XSTest.

Downloads behaviors, prints goals/categories/splits,
verifies the prompt passthrough works.

Run:
    python -m pytest tests/test_06_safety.py -v -s --tb=short
"""
import pytest
from conftest import SAFETY_TASKS, safety_task_kwargs


@pytest.mark.parametrize("task_name", SAFETY_TASKS)
def test_safety_dataset(task_name):
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import DEFAULT_SAFETY_PROFILES

    profiles = {"greedy": DEFAULT_SAFETY_PROFILES["greedy"]}
    task = TaskRegistry.create(
        task_name, **safety_task_kwargs(task_name, 6, profiles),
    )
    items = list(task.iter_items())

    print(f"\n{'='*72}")
    print(f"  SAFETY: {task_name}  source={task.source}")
    print(f"  items: {len(items)}")
    print(f"{'='*72}")

    assert len(items) > 0

    # Collect stats
    splits = {}
    categories = {}
    for item in items:
        splits[item.split] = splits.get(item.split, 0) + 1
        cat = item.category or "(none)"
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Splits: {splits}")
    print(f"  Categories (first 10): {dict(list(categories.items())[:10])}")

    # Print first 3 items
    for i, item in enumerate(items[:3]):
        rendered = task.render_prompt(item)
        print(f"\n  ─── item[{i}] ───")
        print(f"    sample_id      = {item.sample_id}")
        print(f"    behavior_id    = {item.behavior_id}")
        print(f"    split          = {item.split}")
        print(f"    category       = {item.category}")
        print(f"    prompt_variant = {item.prompt_variant}")
        print(f"    profile        = {item.profile}")
        print(f"    prompt_text    = {item.prompt_text[:150]}…")
        print(f"    rendered       = {rendered[:150]}…")
        print(f"    ground_truth   = {item.ground_truth}")

        # Prompt passthrough
        assert rendered == item.prompt_text, "Safety prompt should be raw passthrough"
        assert item.prompt_text.strip(), "Prompt should not be empty"
        assert item.behavior_id, "Must have behavior_id"


@pytest.mark.parametrize("task_name", SAFETY_TASKS)
def test_safety_spec(task_name):
    """Verify SafetySpec is generated correctly for manifest."""
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import DEFAULT_SAFETY_PROFILES

    profiles = {"greedy": DEFAULT_SAFETY_PROFILES["greedy"]}
    task = TaskRegistry.create(
        task_name, **safety_task_kwargs(task_name, 2, profiles),
    )
    spec = task.get_safety_spec()
    assert spec is not None
    assert spec.is_safety_run

    print(f"\n  SafetySpec for {task_name}:")
    print(f"    dataset:      {spec.safety_dataset}")
    print(f"    splits:       {spec.splits}")
    print(f"    variants:     {spec.prompt_variant_types}")
    print(f"    profiles:     {list(spec.profiles.keys())}")