from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import hashlib
import re
import warnings

ANSWER_PREFIXES: Dict[str, str] = {
    'en': 'Answer',
    'bn': 'উত্তর',
    'de': 'Antwort',
    'es': 'Respuesta',
    'fr': 'Réponse',
    'ja': '答え',
    'ko': '답',
    'ru': 'Ответ',
    'sw': 'Jibu',
    'te': 'సమాధానం',
    'th': 'คำตอบ',
    'zh': '答案',
    'pt': 'Resposta',
    'it': 'Risposta',
    'ar': 'الإجابة',
    'hi': 'उत्तर',
    'vi': 'Câu trả lời',
    'tr': 'Cevap',
}


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str
    variables: tuple = ()
    description: str = ''
    answer_prefix: str = 'Answer'
    language: str = 'en'
    requested_language: str = 'en'
    localization_mode: str = 'base'
    supports_multilingual: bool = False

    def __post_init__(self) -> None:
        if not self.variables:
            detected = tuple(dict.fromkeys(re.findall(r'\{(\w+)\}', self.template)))
            object.__setattr__(self, 'variables', detected)
        if not self.requested_language:
            object.__setattr__(self, 'requested_language', self.language)

    def format(self, **kwargs: Any) -> str:
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f'Missing required variables: {missing}')
        return self.template.format(**kwargs)

    def format_safe(self, **kwargs: Any) -> str:
        safe = {name: kwargs.get(name, '') for name in self.variables}
        return self.template.format(**safe)

    def with_language(self, language: str) -> 'PromptTemplate':
        target_language = str(language or self.language)
        if target_language == self.language:
            return PromptTemplate(
                name=self.name,
                template=self.template,
                variables=self.variables,
                description=self.description,
                answer_prefix=self.answer_prefix,
                language=self.language,
                requested_language=target_language,
                localization_mode='base',
                supports_multilingual=self.supports_multilingual,
            )
        if not self.supports_multilingual:
            return PromptTemplate(
                name=self.name,
                template=self.template,
                variables=self.variables,
                description=self.description,
                answer_prefix=self.answer_prefix,
                language=self.language,
                requested_language=target_language,
                localization_mode='requested_only',
                supports_multilingual=self.supports_multilingual,
            )
        current_prefix = ANSWER_PREFIXES.get(self.language, ANSWER_PREFIXES['en'])
        localized_prefix = ANSWER_PREFIXES.get(target_language, ANSWER_PREFIXES['en'])
        new_template = self.template
        replaced = False
        replacements = [
            (current_prefix + ':', localized_prefix + ':'),
            (f'"{current_prefix}:"', f'"{localized_prefix}:"'),
            (f"'{current_prefix}:'", f"'{localized_prefix}:'"),
            (f'"{current_prefix}"', f'"{localized_prefix}"'),
            (f"'{current_prefix}'", f"'{localized_prefix}'"),
        ]
        for old, new in replacements:
            if old in new_template:
                new_template = new_template.replace(old, new)
                replaced = True
        new_answer_prefix = localized_prefix if self.answer_prefix == current_prefix else self.answer_prefix
        return PromptTemplate(
            name=self.name,
            template=new_template,
            variables=self.variables,
            description=self.description,
            answer_prefix=new_answer_prefix,
            language=self.language,
            requested_language=target_language,
            localization_mode='answer_prefix' if replaced else 'requested_only',
            supports_multilingual=self.supports_multilingual,
        )

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.template.encode('utf-8')).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'template': self.template,
            'variables': list(self.variables),
            'description': self.description,
            'answer_prefix': self.answer_prefix,
            'language': self.language,
            'requested_language': self.requested_language,
            'localization_mode': self.localization_mode,
            'supports_multilingual': self.supports_multilingual,
            'hash': self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        variables = tuple(data.get('variables') or ())
        return cls(
            name=str(data['name']),
            template=str(data['template']),
            variables=variables,
            description=str(data.get('description', '')),
            answer_prefix=str(data.get('answer_prefix', 'Answer')),
            language=str(data.get('language', 'en')),
            requested_language=str(data.get('requested_language', data.get('language', 'en'))),
            localization_mode=str(data.get('localization_mode', 'base')),
            supports_multilingual=bool(data.get('supports_multilingual', False)),
        )


_TEMPLATE_MAP: Dict[str, Dict[str, PromptTemplate]] = {}
_LAZY_LOADERS: Dict[str, Callable[[], None]] = {}
_LAZY_LOADED: set = set()
_PROMPTS_DIR = Path(__file__).parent / 'prompts'


def _ensure_loaded(task_name: str) -> None:
    task_lower = task_name.lower()
    if task_lower not in _TEMPLATE_MAP and task_lower in _LAZY_LOADERS and task_lower not in _LAZY_LOADED:
        _LAZY_LOADED.add(task_lower)
        _LAZY_LOADERS[task_lower]()


def register_templates(task_name: str, templates: Dict[str, PromptTemplate], overwrite: bool = False) -> None:
    task_lower = task_name.lower()
    if overwrite or task_lower not in _TEMPLATE_MAP:
        _TEMPLATE_MAP[task_lower] = dict(templates)
        return
    _TEMPLATE_MAP[task_lower].update(templates)


def register_lazy_loader(task_name: str, loader: Callable[[], None]) -> None:
    _LAZY_LOADERS[task_name.lower()] = loader


def get_template(task_name: str, variant: str = 'zot', language: str = 'en') -> PromptTemplate:
    _ensure_loaded(task_name)
    task_templates = _TEMPLATE_MAP.get(task_name.lower())
    if task_templates is None:
        available = ', '.join(sorted(_TEMPLATE_MAP.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")
    template = task_templates.get(variant)
    if template is None:
        available = ', '.join(sorted(task_templates.keys()))
        raise ValueError(f"Unknown variant '{variant}' for task '{task_name}'. Available: {available}")
    return template.with_language(language)


def get_templates(task_name: str) -> Dict[str, PromptTemplate]:
    _ensure_loaded(task_name)
    task_templates = _TEMPLATE_MAP.get(task_name.lower())
    if task_templates is None:
        available = ', '.join(sorted(_TEMPLATE_MAP.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")
    return dict(task_templates)


def list_tasks() -> List[str]:
    return sorted(set(_TEMPLATE_MAP.keys()) | set(_LAZY_LOADERS.keys()))


def has_template(task_name: str) -> bool:
    _ensure_loaded(task_name)
    return task_name.lower() in _TEMPLATE_MAP


def _load_yaml_file(yaml_path: Path) -> Optional[dict]:
    try:
        import yaml
    except ImportError:
        warnings.warn('PyYAML not installed; prompt templates will not be loaded from YAML files. Install with: pip install pyyaml', ImportWarning, stacklevel=3)
        return None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        warnings.warn(f'Failed to load prompt YAML {yaml_path}: {exc}', UserWarning, stacklevel=3)
        return None


def load_templates_from_yaml(prompts_dir: Optional[Path] = None, overwrite: bool = False) -> int:
    prompts_dir = prompts_dir or _PROMPTS_DIR
    if not prompts_dir.is_dir():
        return 0
    loaded = 0
    for yaml_path in sorted(prompts_dir.glob('*.yaml')):
        data = _load_yaml_file(yaml_path)
        if data is None:
            continue
        task_name = yaml_path.stem
        variants_data = data.get('variants', {})
        if not variants_data:
            continue
        templates: Dict[str, PromptTemplate] = {}
        for variant_name, info in variants_data.items():
            template_text = str(info.get('template', ''))
            if template_text.endswith('\n'):
                template_text = template_text[:-1]
            templates[variant_name] = PromptTemplate(
                name=f'{task_name}_{variant_name}',
                template=template_text,
                answer_prefix=str(info.get('answer_prefix', 'Answer')),
                description=str(info.get('description', '')),
                language=str(info.get('language', 'en')),
                requested_language=str(info.get('requested_language', info.get('language', 'en'))),
                localization_mode=str(info.get('localization_mode', 'base')),
                supports_multilingual=bool(info.get('supports_multilingual', False)),
            )
        register_templates(task_name, templates, overwrite=overwrite)
        loaded += 1
    return loaded


load_templates_from_dir = load_templates_from_yaml


def _bootstrap() -> None:
    loaded = load_templates_from_yaml()
    if loaded == 0:
        warnings.warn(f'No prompt YAML files loaded from {_PROMPTS_DIR}. Templates will be empty until explicitly registered.', UserWarning, stacklevel=2)


_bootstrap()
