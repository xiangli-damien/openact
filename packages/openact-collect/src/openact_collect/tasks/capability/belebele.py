from typing import Iterator, Optional

from openact_collect.data import HFDatasetSpec
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry
BELEBELE_LANG_MAP = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "te": "tel_Telu",
    "tr": "tur_Latn",
}
@TaskRegistry.register("belebele")
class BelebeleTask(Task):
    task_name = "belebele"
    source = "facebook/belebele"
    split = "test"
    language = "en"
    default_template = "zot"
    def __init__(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        template: str = "zot",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(
            max_samples=max_samples, split=split, template=template, **kwargs
        )
        self.language = language
        self._lang_code = BELEBELE_LANG_MAP.get(language, language)
        self._dataset = None
    def _load_dataset(self):
        if self._dataset is None:
            self._dataset = self.load_hf_dataset(
                HFDatasetSpec(name="facebook/belebele", config=self._lang_code, split=self.split)
            )

    def estimate_size(self) -> Optional[int]:
        self._load_dataset()
        try:
            n = len(self._dataset)
        except Exception:
            return None
        return min(n, self.max_samples) if self.max_samples else n
    def iter_items(self) -> Iterator[TaskItem]:
        self._load_dataset()
        for idx, item in enumerate(self._dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            passage = item["flores_passage"]
            question = item["question"]
            choices = [
                item["mc_answer1"],
                item["mc_answer2"],
                item["mc_answer3"],
                item["mc_answer4"],
            ]
            correct_num = int(item["correct_answer_num"])
            answer_letter = ["A", "B", "C", "D"][correct_num - 1]
            if self._template_variant == "zot":
                question_full = (
                    f"{passage}\n\nQuestion: {question}\n\n"
                    f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
                )
                prompt_fields = {"question": question_full}
            else:
                prompt_fields = {
                    "passage": passage,
                    "question": question,
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2],
                    "choice_d": choices[3],
                }
            yield TaskItem(
                sample_idx=idx,
                sample_id=f"belebele_{self._lang_code}_{idx}",
                prompt_fields=prompt_fields,
                ground_truth=answer_letter,
                language=self.language,
                meta={
                    "passage": passage,
                    "question": question,
                    "choices": choices,
                    "correct_answer_num": correct_num,
                    "language": self.language,
                    "lang_code": self._lang_code,
                },
            )
