"""04 — Prompt rendering for every task × language you plan to collect.

Shows:
  1. Raw template text
  2. Rendered prompt for item[0] — exactly what the model will see
  3. No unresolved {placeholders}
  4. Template variables match what the task provides

Run:
    python -m pytest tests/test_04_prompt_rendering.py -v -s --tb=short
    python -m pytest tests/test_04_prompt_rendering.py -v -s -k mgsm
"""
import re
import pytest
from conftest import MONO_TASKS, MGSM_LANGS, BELEBELE_LANGS, safety_task_kwargs


def _box(title, text, width=76):
    print(f"\n  ┌{'─'*width}┐")
    print(f"  │ {title:<{width-2}} │")
    print(f"  ├{'─'*width}┤")
    for line in text.split("\n"):
        chunks = [line[i:i+width-4] for i in range(0, max(1, len(line)), width-4)]
        for c in chunks:
            print(f"  │ {c:<{width-2}} │")
    print(f"  └{'─'*width}┘")


def _run_prompt_check(task_name, extra_kwargs=None):
    from openact_collect.tasks.registry import TaskRegistry

    extra = extra_kwargs or {}
    task = TaskRegistry.create(task_name, max_samples=2, **extra)
    items = list(task.iter_items())
    assert items, f"No items for {task_name}"

    tpl = task.get_prompt_template()
    lang = extra.get("language", "en")
    label = f"{task_name}_{lang}" if "language" in extra else task_name

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  template_name={tpl.name}  vars={tpl.variables}  prefix={tpl.answer_prefix!r}  lang={tpl.language}")
    print(f"{'='*80}")

    _box(f"RAW TEMPLATE", tpl.template)

    item = items[0]
    rendered = task.render_prompt(item)
    _box(f"RENDERED (item[0], GT={item.ground_truth})", rendered)

    # No unresolved template variables
    unresolved = [u for u in re.findall(r"\{(\w+)\}", rendered) if u in tpl.variables]
    assert not unresolved, f"Unresolved vars in rendered prompt: {unresolved}"

    assert len(rendered) > 10, f"Prompt too short: {len(rendered)} chars"
    return rendered


# ── Mono tasks ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("task_name", MONO_TASKS)
def test_prompt_mono(task_name):
    _run_prompt_check(task_name)


# ── MGSM all languages ───────────────────────────────────────────────────
@pytest.mark.parametrize("lang", MGSM_LANGS)
def test_prompt_mgsm(lang):
    from openact_core.tasks.templates import ANSWER_PREFIXES

    rendered = _run_prompt_check("mgsm", {"language": lang})
    expected_prefix = ANSWER_PREFIXES.get(lang, "Answer")
    assert expected_prefix in rendered, (
        f"MGSM {lang}: rendered prompt missing localized prefix {expected_prefix!r}"
    )


# ── Belebele all languages ───────────────────────────────────────────────
@pytest.mark.parametrize("lang", BELEBELE_LANGS)
def test_prompt_belebele(lang):
    _run_prompt_check("belebele", {"language": lang})


# ── Safety tasks ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("task_name", ["jbb", "advbench", "xstest"])
def test_prompt_safety(task_name):
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import DEFAULT_SAFETY_PROFILES

    profiles = {"greedy": DEFAULT_SAFETY_PROFILES["greedy"]}
    task = TaskRegistry.create(
        task_name, **safety_task_kwargs(task_name, 2, profiles),
    )
    items = list(task.iter_items())
    assert items

    item = items[0]
    rendered = task.render_prompt(item)
    tpl = task.get_prompt_template()

    print(f"\n  SAFETY: {task_name}")
    print(f"    template = {tpl.template!r}")
    print(f"    prompt_text = {item.prompt_text[:150]}")
    print(f"    rendered    = {rendered[:150]}")

    # Safety = raw passthrough
    assert rendered == item.prompt_text, "Safety prompt should be raw passthrough"