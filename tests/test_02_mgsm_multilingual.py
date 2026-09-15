"""02 — MGSM: all 11 languages.

Downloads each language variant, prints questions and GT,
verifies numeric ground_truth and localized answer prefix.

Run:
    python -m pytest tests/test_02_mgsm_multilingual.py -v -s --tb=short
    python -m pytest tests/test_02_mgsm_multilingual.py -v -s -k ja   # one lang
"""
import pytest
from tests.conftest import MGSM_LANGS
from openact_core.tasks.templates import ANSWER_PREFIXES


@pytest.mark.parametrize("lang", MGSM_LANGS)
def test_mgsm_language(lang):
    from openact_collect.tasks.registry import TaskRegistry

    task = TaskRegistry.create("mgsm", max_samples=3, language=lang)
    items = list(task.iter_items())

    expected_prefix = ANSWER_PREFIXES.get(lang, "Answer")
    tpl = task.get_prompt_template()

    print(f"\n{'='*72}")
    print(f"  MGSM — lang={lang}  ({len(items)} items)")
    print(f"  answer_prefix: {tpl.answer_prefix!r} (expected: {expected_prefix!r})")
    print(f"{'='*72}")

    assert len(items) > 0, f"No items for MGSM {lang}"
    assert tpl.answer_prefix == expected_prefix, (
        f"Wrong prefix: got {tpl.answer_prefix!r}, want {expected_prefix!r}"
    )

    for i, item in enumerate(items[:2]):
        rendered = task.render_prompt(item)
        print(f"\n  ─── item[{i}] ───")
        print(f"    question     = {str(item.prompt_fields.get('question', ''))[:120]}")
        print(f"    ground_truth = {item.ground_truth}")
        print(f"    answer_number= {item.meta.get('answer_number')}")
        print(f"    rendered prompt (first 300 chars):")
        print(f"      {rendered[:300]}")

        # GT should be numeric
        assert item.ground_truth is not None, f"MGSM {lang} GT is None"
        try:
            float(str(item.ground_truth).replace(",", ""))
        except ValueError:
            pytest.fail(f"MGSM {lang} GT not numeric: {item.ground_truth!r}")

        # Rendered prompt should contain the answer prefix
        assert expected_prefix in rendered, (
            f"Rendered prompt missing answer prefix {expected_prefix!r}"
        )

    print(f"\n  ✓ MGSM {lang}: {len(items)} items, numeric GT, prefix correct")