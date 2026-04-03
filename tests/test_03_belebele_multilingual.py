"""03 — Belebele: all 18 languages.

Downloads each language variant, prints passage/question/choices and GT.
Verifies GT is one of A/B/C/D.

Run:
    python -m pytest tests/test_03_belebele_multilingual.py -v -s --tb=short
    python -m pytest tests/test_03_belebele_multilingual.py -v -s -k zh
"""
import pytest
from conftest import BELEBELE_LANGS


# Internal lang code map from the task source
BELEBELE_LANG_MAP = {
    "en": "eng_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab",
    "hi": "hin_Deva", "ko": "kor_Hang", "pt": "por_Latn", "it": "ita_Latn",
    "th": "tha_Thai", "vi": "vie_Latn", "bn": "ben_Beng", "sw": "swh_Latn",
    "te": "tel_Telu", "tr": "tur_Latn",
}


@pytest.mark.parametrize("lang", BELEBELE_LANGS)
def test_belebele_language(lang):
    from openact_collect.tasks.registry import TaskRegistry

    lang_code = BELEBELE_LANG_MAP.get(lang, lang)
    task = TaskRegistry.create("belebele", max_samples=3, language=lang)
    items = list(task.iter_items())

    print(f"\n{'='*72}")
    print(f"  BELEBELE — lang={lang} (HF config={lang_code})  ({len(items)} items)")
    print(f"{'='*72}")

    assert len(items) > 0, f"No items for Belebele {lang}"

    for i, item in enumerate(items[:2]):
        rendered = task.render_prompt(item)
        print(f"\n  ─── item[{i}] ───")
        print(f"    ground_truth = {item.ground_truth}")
        print(f"    correct_answer_num = {item.meta.get('correct_answer_num')}")
        print(f"    passage (first 120) = {str(item.meta.get('passage', ''))[:120]}")
        print(f"    question = {str(item.meta.get('question', ''))[:120]}")
        choices = item.meta.get("choices", [])
        for ci, ch in enumerate(choices):
            label = "ABCD"[ci] if ci < 4 else "?"
            print(f"    choice {label} = {str(ch)[:80]}")
        print(f"    rendered prompt (first 300):")
        print(f"      {rendered[:300]}")

        # GT must be A/B/C/D
        assert item.ground_truth in ("A", "B", "C", "D"), (
            f"Belebele {lang} GT not A-D: {item.ground_truth!r}"
        )

        # GT should correspond to correct_answer_num
        expected_letter = "ABCD"[int(item.meta["correct_answer_num"]) - 1]
        assert item.ground_truth == expected_letter, (
            f"GT {item.ground_truth} != expected {expected_letter} from correct_answer_num={item.meta['correct_answer_num']}"
        )

    print(f"\n  ✓ Belebele {lang}: {len(items)} items, GT A-D verified")