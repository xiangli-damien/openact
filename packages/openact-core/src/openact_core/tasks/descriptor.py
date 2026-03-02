from dataclasses import dataclass
from typing import Dict, List, Optional

EVAL_PARSER = "parser"
EVAL_LLM_JUDGE = "llm_judge"
EVAL_SAFETY = "safety"


@dataclass(frozen=True)
class TaskDescriptor:
    name: str
    source: str
    parser_type: str
    matcher_type: str = "exact"
    eval_strategy: str = EVAL_PARSER
    language: str = "en"
    default_split: str = "test"
    default_template: str = "cot"
    available_templates: tuple = ()
    task_type: str = "benchmark"
    domain: str = "general"
    is_safety: bool = False


TASK_DESCRIPTORS: Dict[str, TaskDescriptor] = {
    "gsm8k": TaskDescriptor(
        name="gsm8k",
        source="openai/gsm8k",
        parser_type="numeric",
        matcher_type="numeric",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "direct", "simple"),
        domain="math",
    ),
    "mgsm": TaskDescriptor(
        name="mgsm",
        source="juletxara/mgsm",
        parser_type="numeric",
        matcher_type="numeric",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "cot_native", "direct", "simple"),
        domain="math",
    ),
    "math": TaskDescriptor(
        name="math",
        source="EleutherAI/hendrycks_math",
        parser_type="math",
        matcher_type="math",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "boxed", "simple"),
        domain="math",
    ),
    "theoremqa": TaskDescriptor(
        name="theoremqa",
        source="TIGER-Lab/TheoremQA",
        parser_type="theoremqa",
        matcher_type="type_aware",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "simple"),
        domain="math",
    ),
    "mmlu": TaskDescriptor(
        name="mmlu",
        source="cais/mmlu",
        parser_type="mc4",
        matcher_type="exact",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "simple"),
        domain="knowledge",
    ),
    "arc_challenge": TaskDescriptor(
        name="arc_challenge",
        source="allenai/ai2_arc",
        parser_type="mc5",
        matcher_type="exact",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "simple"),
        domain="reasoning",
    ),
    "commonsenseqa": TaskDescriptor(
        name="commonsenseqa",
        source="tau/commonsense_qa",
        parser_type="mc5",
        matcher_type="exact",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "simple"),
        domain="reasoning",
    ),
    "belebele": TaskDescriptor(
        name="belebele",
        source="facebook/belebele",
        parser_type="mc4",
        matcher_type="exact",
        eval_strategy=EVAL_PARSER,
        default_template="cot",
        available_templates=("cot", "simple"),
        domain="knowledge",
    ),
    "truthfulqa": TaskDescriptor(
        name="truthfulqa",
        source="truthful_qa",
        parser_type="freeform",
        matcher_type="exact",
        eval_strategy=EVAL_LLM_JUDGE,
        default_split="validation",
        default_template="cot",
        available_templates=("cot", "simple"),
        domain="reasoning",
    ),
    "humaneval": TaskDescriptor(
        name="humaneval",
        source="openai/openai_humaneval",
        parser_type="code",
        eval_strategy=EVAL_LLM_JUDGE,
        default_template="instruct",
        available_templates=("instruct", "raw"),
        task_type="code",
        domain="code",
    ),
    "ifeval": TaskDescriptor(
        name="ifeval",
        source="google/IFEval",
        parser_type="freeform",
        eval_strategy=EVAL_LLM_JUDGE,
        default_template="raw",
        available_templates=("raw", "instruct"),
        task_type="freeform",
        domain="reasoning",
    ),
    "jbb": TaskDescriptor(
        name="jbb",
        source="JailbreakBench/JBB-Behaviors",
        parser_type="refusal",
        eval_strategy=EVAL_SAFETY,
        default_template="raw",
        available_templates=("raw", "system_context"),
        task_type="safety",
        is_safety=True,
        domain="safety",
    ),
    "advbench": TaskDescriptor(
        name="advbench",
        source="walledai/AdvBench",
        parser_type="refusal",
        eval_strategy=EVAL_SAFETY,
        default_template="raw",
        available_templates=("raw", "system_context"),
        task_type="safety",
        is_safety=True,
        domain="safety",
    ),
    "xstest": TaskDescriptor(
        name="xstest",
        source="nreimers/XSTest",
        parser_type="refusal",
        eval_strategy=EVAL_SAFETY,
        default_template="raw",
        available_templates=("raw", "system_context"),
        task_type="safety",
        is_safety=True,
        domain="safety",
    ),
}


def get_descriptor(task_name: str) -> TaskDescriptor:
    d = TASK_DESCRIPTORS.get(task_name.lower())
    if d is None:
        available = ", ".join(sorted(TASK_DESCRIPTORS.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")
    return d


def has_descriptor(task_name: str) -> bool:
    return task_name.lower() in TASK_DESCRIPTORS


def list_tasks_by_domain(domain: str) -> List[str]:
    return [name for name, d in TASK_DESCRIPTORS.items() if d.domain == domain]


def list_tasks_by_type(task_type: str) -> List[str]:
    return [name for name, d in TASK_DESCRIPTORS.items() if d.task_type == task_type]


def register_task(descriptor: TaskDescriptor) -> None:
    TASK_DESCRIPTORS[descriptor.name.lower()] = descriptor


def get_registered_template_names(task_name: str) -> List[str]:
    from openact_core.tasks.templates import get_available_template_names

    return get_available_template_names(task_name)