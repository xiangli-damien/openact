from openact_core.tasks.parsers.base import (
    AnswerParser,
    normalize_numeric,
    normalize_letter,
    normalize_text,
    normalize_math_expr,
    extract_boxed,
    find_last_number,
)
from openact_core.tasks.parsers.numeric import NumericParser
from openact_core.tasks.parsers.multiple_choice import (
    MultipleChoiceParser,
    MCParser4,
    MCParser5,
)
from openact_core.tasks.parsers.math_parser import MathParser
from openact_core.tasks.parsers.theoremqa import TheoremQAParser
from openact_core.tasks.parsers.code import CodeParser
from openact_core.tasks.parsers.freeform import TruthfulQAParser, IFEvalParser
from openact_core.tasks.parsers.safety import RefusalParser, SafetyLabelParser
__all__ = [
    "AnswerParser",
    "normalize_numeric",
    "normalize_letter",
    "normalize_text",
    "normalize_math_expr",
    "extract_boxed",
    "find_last_number",
    "NumericParser",
    "MultipleChoiceParser",
    "MCParser4",
    "MCParser5",
    "MathParser",
    "TheoremQAParser",
    "CodeParser",
    "TruthfulQAParser",
    "IFEvalParser",
    "get_parser",
    "list_parsers",
]
_PARSER_MAP = {
    "gsm8k": NumericParser,
    "mgsm": NumericParser,
    "numeric": NumericParser,
    "mmlu": MCParser4,
    "belebele": MCParser4,
    "mc4": MCParser4,
    "arc_challenge": MCParser5,
    "commonsenseqa": MCParser5,
    "mc5": MCParser5,
    "math": MathParser,
    "theoremqa": TheoremQAParser,
    "humaneval": CodeParser,
    "code": CodeParser,
    "truthfulqa": TruthfulQAParser,
    "ifeval": IFEvalParser,
    "jbb": RefusalParser,
    "advbench": RefusalParser,
    "xstest": RefusalParser,
}
def get_parser(task_name: str, **kwargs) -> AnswerParser:
    cls = _PARSER_MAP.get(task_name.lower())
    if cls is None:
        available = ", ".join(sorted(_PARSER_MAP.keys()))
        raise ValueError(
            f"Unknown parser '{task_name}'. Available: {available}"
        )
    return cls(**kwargs)
def list_parsers() -> list:
    return sorted(_PARSER_MAP.keys())
