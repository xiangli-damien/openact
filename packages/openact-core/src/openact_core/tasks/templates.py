"""
Prompt template registry for OpenAct tasks.

Templates are loaded from per-task YAML files in the ``prompts/`` directory
at import time.  The YAML schema per file is::

    variants:
      zot:
        template: |
          ...
        answer_prefix: "Answer"
        description: "..."
      simple:
        template: "{question}"
        answer_prefix: "Answer"
        description: "Raw question only."

Each variant is converted to a :class:`PromptTemplate` and registered under
``(task_name, variant_name)``.  Callers retrieve templates via
:func:`get_template`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable
import hashlib
import re
import warnings

# ---------------------------------------------------------------------------
# Multilingual answer prefixes
# ---------------------------------------------------------------------------

ANSWER_PREFIXES: Dict[str, str] = {
    "en": "Answer",
    "bn": "উত্তর",
    "de": "Antwort",
    "es": "Respuesta",
    "fr": "Réponse",
    "ja": "答え",
    "ko": "답",
    "ru": "Ответ",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "zh": "答案",
    "pt": "Resposta",
    "it": "Risposta",
    "ar": "الإجابة",
    "hi": "उत्तर",
    "vi": "Câu trả lời",
    "tr": "Cevap",
}


# ---------------------------------------------------------------------------
# PromptTemplate dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt template with auto-detected variables."""

    name: str
    template: str
    variables: tuple = ()
    description: str = ""
    answer_prefix: str = "Answer"
    language: str = "en"

    def __post_init__(self):
        if not self.variables:
            detected = tuple(
                dict.fromkeys(re.findall(r"\{(\w+)\}", self.template))
            )
            object.__setattr__(self, "variables", detected)

    # -- formatting ----------------------------------------------------------

    def format(self, **kwargs) -> str:
        """Format the template, raising on missing variables."""
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing required variables: {missing}")
        return self.template.format(**kwargs)

    def format_safe(self, **kwargs) -> str:
        """Format the template, substituting '' for missing variables."""
        safe = {v: kwargs.get(v, "") for v in self.variables}
        return self.template.format(**safe)

    # -- multilingual --------------------------------------------------------

    def with_language(self, language: str) -> "PromptTemplate":
        """Return a copy with the answer prefix localized to *language*."""
        if language == self.language:
            return self
        localized_prefix = ANSWER_PREFIXES.get(language, ANSWER_PREFIXES["en"])
        current_prefix = ANSWER_PREFIXES.get(self.language, ANSWER_PREFIXES["en"])

        new_template = self.template
        # Replace bare prefix:  Answer: → 答案:
        if current_prefix + ":" in new_template:
            new_template = new_template.replace(
                current_prefix + ":", localized_prefix + ":"
            )
        # Replace quoted forms:  "Answer:" → "答案:"
        if f'"{current_prefix}:"' in new_template:
            new_template = new_template.replace(
                f'"{current_prefix}:"', f'"{localized_prefix}:"'
            )
        if f'"{current_prefix}"' in new_template:
            new_template = new_template.replace(
                f'"{current_prefix}"', f'"{localized_prefix}"'
            )

        return PromptTemplate(
            name=self.name,
            template=new_template,
            variables=self.variables,
            description=self.description,
            answer_prefix=localized_prefix,
            language=language,
        )

    # -- identity ------------------------------------------------------------

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# task_name (lower) → variant_name → PromptTemplate
_TEMPLATE_MAP: Dict[str, Dict[str, PromptTemplate]] = {}

# Lazy loaders for tasks not shipped as YAML
_LAZY_LOADERS: Dict[str, Callable[[], None]] = {}
_LAZY_LOADED: set = set()


def _ensure_loaded(task_name: str) -> None:
    task_lower = task_name.lower()
    if task_lower not in _TEMPLATE_MAP and task_lower in _LAZY_LOADERS:
        if task_lower not in _LAZY_LOADED:
            _LAZY_LOADED.add(task_lower)
            _LAZY_LOADERS[task_lower]()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_templates(
    task_name: str,
    templates: Dict[str, PromptTemplate],
    overwrite: bool = False,
) -> None:
    """Register a dict of ``{variant: PromptTemplate}`` for *task_name*."""
    task_lower = task_name.lower()
    if overwrite or task_lower not in _TEMPLATE_MAP:
        _TEMPLATE_MAP[task_lower] = templates
    else:
        _TEMPLATE_MAP[task_lower].update(templates)


def register_lazy_loader(task_name: str, loader: Callable[[], None]) -> None:
    """Register a callable that will populate templates on first access."""
    _LAZY_LOADERS[task_name.lower()] = loader


def get_template(
    task_name: str,
    variant: str = "zot",
    language: str = "en",
) -> PromptTemplate:
    """Retrieve a template by task and variant, with optional localization."""
    _ensure_loaded(task_name)
    task_templates = _TEMPLATE_MAP.get(task_name.lower())
    if task_templates is None:
        available = ", ".join(sorted(_TEMPLATE_MAP.keys()))
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {available}"
        )
    tpl = task_templates.get(variant)
    if tpl is None:
        available = ", ".join(sorted(task_templates.keys()))
        raise ValueError(
            f"Unknown variant '{variant}' for task '{task_name}'. "
            f"Available: {available}"
        )
    if language != "en":
        tpl = tpl.with_language(language)
    return tpl


def get_templates(task_name: str) -> Dict[str, PromptTemplate]:
    """Return all variants for a given task."""
    _ensure_loaded(task_name)
    task_templates = _TEMPLATE_MAP.get(task_name.lower())
    if task_templates is None:
        available = ", ".join(sorted(_TEMPLATE_MAP.keys()))
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {available}"
        )
    return dict(task_templates)


def list_tasks() -> List[str]:
    """Return sorted list of all registered task names."""
    all_tasks = set(_TEMPLATE_MAP.keys()) | set(_LAZY_LOADERS.keys())
    return sorted(all_tasks)


def has_template(task_name: str) -> bool:
    _ensure_loaded(task_name)
    return task_name.lower() in _TEMPLATE_MAP


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_yaml_file(yaml_path: Path) -> Optional[dict]:
    """Load a single YAML file.  Returns None on failure."""
    try:
        import yaml
    except ImportError:
        warnings.warn(
            "PyYAML not installed; prompt templates will not be loaded "
            "from YAML files.  Install with: pip install pyyaml",
            ImportWarning,
            stacklevel=3,
        )
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to load prompt YAML {yaml_path}: {exc}",
            UserWarning,
            stacklevel=3,
        )
        return None


def load_templates_from_yaml(
    prompts_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> int:
    """
    Scan *prompts_dir* for ``*.yaml`` files and register each as a task.

    Returns the number of tasks successfully loaded.
    """
    prompts_dir = prompts_dir or _PROMPTS_DIR
    if not prompts_dir.is_dir():
        return 0

    loaded = 0
    for yaml_path in sorted(prompts_dir.glob("*.yaml")):
        data = _load_yaml_file(yaml_path)
        if data is None:
            continue

        task_name = yaml_path.stem  # e.g. "gsm8k"
        variants_data = data.get("variants", {})
        if not variants_data:
            continue

        templates: Dict[str, PromptTemplate] = {}
        for variant_name, vinfo in variants_data.items():
            template_text = vinfo.get("template", "")
            # Strip single trailing newline that YAML literal blocks add
            if template_text.endswith("\n"):
                template_text = template_text[:-1]

            templates[variant_name] = PromptTemplate(
                name=f"{task_name}_{variant_name}",
                template=template_text,
                answer_prefix=vinfo.get("answer_prefix", "Answer"),
                description=vinfo.get("description", ""),
            )

        register_templates(task_name, templates, overwrite=overwrite)
        loaded += 1

    return loaded


# Alias for public API (openact_core.tasks imports this name).
load_templates_from_dir = load_templates_from_yaml


# ---------------------------------------------------------------------------
# Bootstrap: load YAML templates at import time
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    n = load_templates_from_yaml()
    if n == 0:
        warnings.warn(
            f"No prompt YAML files loaded from {_PROMPTS_DIR}. "
            "Templates will be empty until explicitly registered.",
            UserWarning,
            stacklevel=2,
        )


_bootstrap()