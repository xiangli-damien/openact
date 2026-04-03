"""07 — End-to-end simulation: dataset → prompt → (fake response) → parser → matcher.

For each task, this simulates the full collection+eval pipeline:
  1. Download dataset item
  2. Render prompt (what model sees)
  3. Construct a synthetic correct response
  4. Parse the response
  5. Match against ground truth

If this passes, the real pipeline should produce correct labels.

Run:
    python -m pytest tests/test_07_end_to_end_sim.py -v -s --tb=short
"""
import pytest
from openact_core.tasks.parsers import get_parser
from openact_eval.matchers import get_matcher


def _make_correct_response(task_name, gt):
    """Build a synthetic model response that contains the correct answer."""
    if gt is None:
        return "Here is my detailed response..."

    if task_name in ("gsm8k", "mgsm"):
        return f"Let me solve step by step.\nFirst calculate 10 + 5 = 15.\nThen multiply by 2 = 30.\nAnswer: {gt}"

    if task_name in ("mmlu", "arc_challenge", "commonsenseqa", "belebele"):
        return f"After careful analysis, the correct choice is ({gt}) because it best fits.\nAnswer: {gt}"

    if task_name == "math":
        return f"Using algebra, we get $\\boxed{{{gt}}}$."

    if task_name == "theoremqa":
        return f"Applying the theorem, we find the answer is {gt}.\nAnswer: {gt}"

    if task_name == "truthfulqa":
        return f"The truthful answer is: {gt}"

    return f"Answer: {gt}"


# ── Mono tasks ────────────────────────────────────────────────────────────
_MONO_PARSEABLE = [
    "gsm8k", "mmlu", "math", "theoremqa",
    "arc_challenge", "commonsenseqa", "belebele",
]


@pytest.mark.parametrize("task_name", _MONO_PARSEABLE)
def test_e2e_mono(task_name):
    from openact_collect.tasks.registry import TaskRegistry

    extra = {}
    if task_name == "belebele":
        extra["language"] = "en"

    task = TaskRegistry.create(task_name, max_samples=3, **extra)
    items = list(task.iter_items())
    assert items

    parser = get_parser(task_name)
    matcher = get_matcher(task_name)

    print(f"\n{'='*72}")
    print(f"  E2E: {task_name}")
    print(f"{'='*72}")

    for item in items[:3]:
        gt = item.ground_truth
        prompt = task.render_prompt(item)
        response = _make_correct_response(task_name, gt)

        extracted = parser.extract(response)
        normalized = parser.normalize(extracted)
        gt_norm = parser.normalize(gt) if gt else None

        is_match = matcher.match(normalized, gt_norm) if gt_norm else None

        print(f"\n    idx={item.sample_idx}")
        print(f"    prompt (first 100):  {prompt[:100]}…")
        print(f"    GT:                  {gt}")
        print(f"    fake response:       {response[:100]}…")
        print(f"    extracted:           {extracted!r}")
        print(f"    normalized:          {normalized!r}")
        print(f"    GT normalized:       {gt_norm!r}")
        print(f"    match:               {is_match}")

        if gt is not None:
            assert is_match is True, (
                f"E2E failed for {task_name} item {item.sample_idx}: "
                f"extracted={normalized!r} vs gt={gt_norm!r}"
            )

    print(f"\n  ✓ {task_name}: all items extracted and matched correctly")


# ── MGSM selected languages ──────────────────────────────────────────────
@pytest.mark.parametrize("lang", ["en", "zh", "ja", "de", "fr"])
def test_e2e_mgsm(lang):
    from openact_collect.tasks.registry import TaskRegistry

    task = TaskRegistry.create("mgsm", max_samples=2, language=lang)
    items = list(task.iter_items())
    assert items

    parser = get_parser("mgsm")
    matcher = get_matcher("mgsm")

    print(f"\n  E2E MGSM {lang}:")
    for item in items[:2]:
        gt = item.ground_truth
        response = f"計算すると {gt} です。\nAnswer: {gt}"  # mix of langs, parser should find number
        extracted = parser.extract(response)
        normalized = parser.normalize(extracted)
        gt_norm = parser.normalize(gt)
        is_match = matcher.match(normalized, gt_norm)

        print(f"    GT={gt}, extracted={extracted!r}, match={is_match}")
        assert is_match, f"MGSM {lang}: {normalized!r} != {gt_norm!r}"


# ── Belebele selected languages ──────────────────────────────────────────
@pytest.mark.parametrize("lang", ["en", "zh", "ja", "ar", "ko"])
def test_e2e_belebele(lang):
    from openact_collect.tasks.registry import TaskRegistry

    task = TaskRegistry.create("belebele", max_samples=2, language=lang)
    items = list(task.iter_items())
    assert items

    parser = get_parser("belebele")
    matcher = get_matcher("belebele")

    print(f"\n  E2E Belebele {lang}:")
    for item in items[:2]:
        gt = item.ground_truth
        response = f"The answer is ({gt})."
        extracted = parser.extract(response)
        normalized = parser.normalize(extracted)
        gt_norm = parser.normalize(gt)
        is_match = matcher.match(normalized, gt_norm)

        print(f"    GT={gt}, extracted={extracted!r}, match={is_match}")
        assert is_match, f"Belebele {lang}: {normalized!r} != {gt_norm!r}"