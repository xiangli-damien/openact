"""
MMLU data-loading task.

Loads from ``cais/mmlu`` and yields formatted ``TaskItem`` objects.
"""

from typing import Iterator, Optional, List

from datasets import load_dataset

from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.registry import TaskRegistry


# Full list of MMLU subjects.
ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


@TaskRegistry.register("mmlu")
class MMLUTask(Task):
    task_name = "mmlu"
    source = "cais/mmlu"
    split = "test"
    language = "en"
    default_template = "zot"

    def __init__(
        self,
        subjects: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        max_per_subject: Optional[int] = None,
        split: str = "test",
        template: str = "zot",
        **kwargs,
    ):
        super().__init__(max_samples=max_samples, split=split, template=template, **kwargs)
        self.subjects = subjects or ALL_SUBJECTS
        self.max_per_subject = max_per_subject
        self._datasets = {}

    def _load_subject(self, subject: str):
        if subject not in self._datasets:
            self._datasets[subject] = load_dataset(
                "cais/mmlu", subject, split=self.split
            )

    def iter_items(self) -> Iterator[TaskItem]:
        template = self.get_prompt_template()
        total_idx = 0

        for subject in self.subjects:
            self._load_subject(subject)
            dataset = self._datasets[subject]
            subject_display = subject.replace("_", " ").title()
            subject_count = 0

            for item in dataset:
                if self.max_samples and total_idx >= self.max_samples:
                    return
                if self.max_per_subject and subject_count >= self.max_per_subject:
                    break

                choices = item["choices"]
                if len(choices) != 4:
                    continue

                if self._template_variant == "zot":
                    question_full = (
                        f"{item['question']}\n\n"
                        f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
                    )
                    prompt_text = template.format_safe(question=question_full)
                else:
                    prompt_kwargs = {
                        "subject": subject_display,
                        "question": item["question"],
                        "choice_a": choices[0],
                        "choice_b": choices[1],
                        "choice_c": choices[2],
                        "choice_d": choices[3],
                    }
                    prompt_text = template.format_safe(**prompt_kwargs)

                answer_idx = item["answer"]
                answer_letter = ["A", "B", "C", "D"][answer_idx]

                yield TaskItem(
                    sample_idx=total_idx,
                    sample_id=f"mmlu_{subject}_{subject_count}",
                    prompt_text=prompt_text,
                    ground_truth=answer_letter,
                    meta={
                        "subject": subject,
                        "question": item["question"],
                        "choices": choices,
                        "answer_idx": answer_idx,
                    },
                )
                total_idx += 1
                subject_count += 1
