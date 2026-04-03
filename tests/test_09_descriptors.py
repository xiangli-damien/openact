"""09 — Task descriptor, eval strategy, and template consistency.

Verifies the wiring between:
  openact-core descriptors ↔ openact-collect tasks ↔ openact-eval evaluators

Run:
    python -m pytest tests/test_09_descriptors.py -v -s --tb=short
"""
import pytest


def test_every_task_has_descriptor():
    """Every registered collect task should have a core descriptor."""
    from openact_collect.tasks.registry import TaskRegistry
    from openact_core.tasks.descriptor import has_descriptor, get_descriptor

    all_tasks = TaskRegistry.list()
    print(f"\n  Checking {len(all_tasks)} tasks against descriptors:")
    for name in all_tasks:
        has = has_descriptor(name)
        if has:
            d = get_descriptor(name)
            print(f"    {name:20s} ✓  eval={d.eval_strategy:10s} parser={d.parser_type:12s} matcher={d.matcher_type:10s} domain={d.domain}")
        else:
            print(f"    {name:20s} ✗  NO DESCRIPTOR")
            # generic/prepared are expected to not have descriptors
            if name not in ("generic", "prepared"):
                pytest.fail(f"Task {name!r} has no descriptor")


def test_every_parser_task_has_parser():
    """Tasks with eval_strategy=parser must have a registered parser (collect tasks only)."""
    from openact_collect.tasks.registry import TaskRegistry
    from openact_core.tasks.descriptor import TASK_DESCRIPTORS, EVAL_PARSER
    from openact_core.tasks.parsers import list_parsers

    parsers = set(list_parsers())
    collect_tasks = set(TaskRegistry.list())
    print(f"\n  Available parsers: {sorted(parsers)}")
    for name, desc in sorted(TASK_DESCRIPTORS.items()):
        if desc.eval_strategy == EVAL_PARSER and name in collect_tasks:
            ok = desc.parser_type in parsers or name in parsers
            print(f"    {name:20s} parser_type={desc.parser_type:12s} {'✓' if ok else '✗'}")
            assert ok, f"Task {name} needs parser {desc.parser_type!r} but it's not registered"


def test_every_task_has_templates():
    """Every registered task should have prompt templates."""
    from openact_core.tasks.templates import has_template, get_templates

    # These tasks use YAML templates from openact-core
    tasks_with_templates = [
        "gsm8k", "mgsm", "mmlu", "math", "theoremqa",
        "arc_challenge", "commonsenseqa", "belebele", "truthfulqa",
        "humaneval", "ifeval",
    ]
    print(f"\n  Template check:")
    for name in tasks_with_templates:
        has = has_template(name)
        if has:
            variants = list(get_templates(name).keys())
            print(f"    {name:20s} ✓  variants={variants}")
        else:
            print(f"    {name:20s} ✗  NO TEMPLATES")
            pytest.fail(f"Task {name!r} has no templates")


def test_eval_config_exists_for_key_tasks():
    """Key tasks should have EvalConfig entries."""
    from openact_eval.eval_descriptor import get_eval_config, EVAL_CONFIGS

    print(f"\n  EvalConfig entries: {sorted(EVAL_CONFIGS.keys())}")
    for name in ["gsm8k", "mgsm", "math", "theoremqa", "truthfulqa", "jbb", "advbench", "xstest"]:
        cfg = get_eval_config(name)
        print(f"    {name:20s} rel_tol={cfg.numeric_rel_tol}, abs_tol={cfg.numeric_abs_tol}")