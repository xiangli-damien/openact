"""05 — Parser & matcher validation.

Feeds synthetic model-like responses through each parser + matcher.
If the parser extracts the wrong thing, you'll collect data with wrong labels.

Run:
    python -m pytest tests/test_05_parsers.py -v -s --tb=short
"""
import pytest
from openact_core.tasks.parsers import get_parser, list_parsers
from openact_eval.matchers import get_matcher


# ──────────────────────────────────────────────────────────────────────────
# (task, response, expected_extracted, gt, should_match)
# ──────────────────────────────────────────────────────────────────────────
CASES = [
    # ── GSM8K ─────────────────────────────────────────────────────────
    ("gsm8k", "Step 1: 5+3=8\nStep 2: 8*2=16\nAnswer: 16", "16", "16", True),
    ("gsm8k", "So the total is $1,234.\n#### 1234", "1234", "1234", True),
    ("gsm8k", "Answer: 42", "42", "45", False),
    ("gsm8k", "The answer is -3.5.\nAnswer: -3.5", "-3.5", "-3.5", True),
    ("gsm8k", "Therefore she has 0 apples.\nAnswer: 0", "0", "0", True),

    # ── MGSM (same numeric parser) ────────────────────────────────────
    ("mgsm", "答案: 150", "150", "150", True),
    ("mgsm", "Antwort: 42", "42", "42", True),
    ("mgsm", "Réponse: 7", "7", "7", True),

    # ── MMLU (mc4: A-D) ──────────────────────────────────────────────
    ("mmlu", "Answer: C", "C", "C", True),
    ("mmlu", "The answer is (B).", "B", "B", True),
    ("mmlu", "I think D is correct.\nAnswer: D", "D", "A", False),
    ("mmlu", "A", "A", "A", True),

    # ── Belebele (mc4: A-D) ──────────────────────────────────────────
    ("belebele", "Based on the passage, the answer is D.", "D", "D", True),
    ("belebele", "(A)", "A", "A", True),

    # ── ARC-Challenge (mc5: A-E) ─────────────────────────────────────
    ("arc_challenge", "Answer: A", "A", "A", True),
    ("arc_challenge", "(E)", "E", "B", False),

    # ── CommonsenseQA (mc5: A-E) ─────────────────────────────────────
    ("commonsenseqa", "Answer: B", "B", "B", True),

    # ── MATH ──────────────────────────────────────────────────────────
    ("math", "$\\boxed{\\frac{3}{5}}$", "\\frac{3}{5}", "\\frac{3}{5}", True),
    ("math", "$\\boxed{42}$", "42", "42", True),
    ("math", "$\\boxed{2\\sqrt{3}}$", "2\\sqrt{3}", "2\\sqrt{3}", True),
    ("math", "The answer is $\\boxed{-7}$.", "-7", "-7", True),
    ("math", "So the answer is 100.\nAnswer: 100", "100", "100", True),

    # ── TheoremQA ─────────────────────────────────────────────────────
    ("theoremqa", "Therefore the answer is 3.14159.\nAnswer: 3.14159", "3.14159", "3.14159", True),
    ("theoremqa", "The answer is True.\nAnswer: True", "True", "True", True),

    # ── HumanEval (code) ──────────────────────────────────────────────
    ("humaneval", "```python\ndef f(x):\n    return x+1\n```", "def f(x):\n    return x+1", None, None),

    # ── Safety (refusal detection) ────────────────────────────────────
    ("jbb", "I cannot help with that request.", "refusal", None, None),
    ("jbb", "Sure! Here is how to do it...", "compliance", None, None),
    ("advbench", "I'm sorry, but I can't assist with that. As an AI...", "refusal", None, None),
    ("advbench", "Here are the steps: 1. First...", "compliance", None, None),
    ("xstest", "I will not provide that information.", "refusal", None, None),
    ("xstest", "Of course! The answer to your question is...", "compliance", None, None),
]


@pytest.mark.parametrize(
    "task,response,expected,gt,should_match",
    CASES,
    ids=[f"{c[0]}_{i}" for i, c in enumerate(CASES)],
)
def test_parser(task, response, expected, gt, should_match):
    parser = get_parser(task)
    extracted = parser.extract(response)
    normalized = parser.normalize(extracted)

    print(f"\n  {task}: {response[:80]}{'…' if len(response)>80 else ''}")
    print(f"    extracted  = {extracted!r}")
    print(f"    normalized = {normalized!r}")

    assert extracted is not None, f"Parser returned None"
    if task in ("jbb", "advbench", "xstest"):
        assert extracted == expected, f"Got {extracted!r}, want {expected!r}"
        return
    if should_match is None:
        return

    matcher = get_matcher(task)
    gt_norm = parser.normalize(gt)
    result = matcher.match(normalized, gt_norm)
    print(f"    gt_norm    = {gt_norm!r}")
    print(f"    match      = {result} (expected {should_match})")
    assert result == should_match


# ── Edge cases ────────────────────────────────────────────────────────────
def test_numeric_parser_edge_cases():
    p = get_parser("gsm8k")
    cases = [
        ("$1,234,567", "1234567"),
        ("-42.5", "-42.5"),
        ("#### 0", "0"),
        ("3.14", "3.14"),
        ("Answer: 15", "15"),
    ]
    print("\n  Numeric edge cases:")
    for resp, expected in cases:
        ext = p.extract(resp)
        norm = p.normalize(ext)
        print(f"    {resp:30s} → norm={norm!r}")
        assert norm == expected, f"Got {norm!r}, want {expected!r}"


def test_math_parser_nested_boxed():
    p = get_parser("math")
    cases = [
        ("$\\boxed{\\frac{1}{2}}$", "\\frac{1}{2}"),
        ("$\\boxed{x^2 + 1}$", "x^2 + 1"),
        ("$\\boxed{\\sqrt{2}}$", "\\sqrt{2}"),
        ("$\\boxed{\\frac{\\sqrt{3}}{2}}$", "\\frac{\\sqrt{3}}{2}"),
    ]
    print("\n  Nested \\boxed{} cases:")
    for resp, expected in cases:
        ext = p.extract(resp)
        print(f"    {resp:45s} → {ext!r}")
        assert ext == expected, f"Got {ext!r}, want {expected!r}"


def test_mc_parser_tricky_formats():
    p4 = get_parser("mc4")
    p5 = get_parser("mc5")
    cases4 = [
        ("The correct option is C because...", "C"),
        ("A. Paris\nB. London\n\nAnswer: A", "A"),
        ("(D) All of the above", "D"),
    ]
    print("\n  MC4 tricky:")
    for resp, exp in cases4:
        ext = p4.extract(resp)
        norm = p4.normalize(ext)
        print(f"    {resp[:50]:50s} → {norm!r}")
        assert norm == exp

    # MC5 should accept E
    ext = p5.extract("Answer: E")
    assert p5.normalize(ext) == "E"
    print(f"    MC5 'Answer: E' → 'E' ✓")


def test_all_parsers_registered():
    parsers = list_parsers()
    print(f"\n  Registered parsers ({len(parsers)}): {parsers}")
    needed = ["gsm8k", "mgsm", "mmlu", "belebele", "math", "theoremqa",
              "arc_challenge", "commonsenseqa", "humaneval", "jbb", "advbench", "xstest"]
    for n in needed:
        assert n in parsers, f"Parser {n!r} not registered"
    print(f"  ✓ All needed parsers present")