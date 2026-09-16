"""Validated TOML configuration for collection and downstream experiment metadata."""
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ImportError:  # Python 3.9/3.10
    import tomli as tomllib

from openact_collect.schema import CaptureSpec, GenerationSpec


def load_run_config(path: str) -> Dict[str, Any]:
    with Path(path).open('rb') as handle:
        config = tomllib.load(handle)
    allowed = {'model', 'collection', 'generation', 'capture', 'analysis', 'model_catalog', 'safety_judge'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
    section_fields = {
        'model': {'identifier', 'revision', 'dtype', 'device_map', 'attn_implementation', 'backend', 'chat_template_kwargs'},
        'collection': {'task', 'output', 'max_samples', 'split', 'template', 'language', 'dataset_revision', 'prepared_path', 'extraction'},
        'capture': set(CaptureSpec.__dataclass_fields__),
        'generation': set(GenerationSpec.__dataclass_fields__),
        'analysis': {'execution', 'device', 'representations', 'normalize_hidden_states', 'seed', 'gmm', 'alignment', 'split', 'smoothing', 'monitoring'},
        'safety_judge': {'identifier', 'revision', 'version_status', 'dtype', 'device_map'},
    }
    for section, fields in section_fields.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be a TOML table')
        unknown = set(values) - fields
        if unknown:
            raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')
    CaptureSpec(**config.get('capture', {}))
    GenerationSpec(**config.get('generation', {}))
    if config.get('model', {}).get('backend', 'transformers') != 'transformers':
        raise ValueError('Only the transformers backend is implemented')
    if config.get('model', {}).get('dtype', 'auto') not in ('auto', 'bfloat16', 'float16', 'float32'):
        raise ValueError('Unsupported model dtype')
    collection = config.get('collection', {})
    if collection.get('extraction', 'teacher_forced_forward') != 'teacher_forced_forward':
        raise ValueError('Extraction must use teacher_forced_forward')
    if collection.get('max_samples', 1) < 1:
        raise ValueError('max_samples must be positive')
    analysis = config.get('analysis', {})
    if analysis:
        if analysis.get('execution') != 'external':
            raise ValueError('Analysis settings are metadata for an external fitter; execution must be external')
        groups = {
            'gmm': {'covariance_type', 'k_selection', 'k_min', 'k_max'},
            'alignment': {'algorithm', 'metric', 'eta'},
            'split': {'stratified', 'fit_fraction', 'evaluation_fraction'},
            'smoothing': {'method', 'alpha'},
            'monitoring': {'target_far'},
        }
        for group, fields in groups.items():
            unknown = set(analysis.get(group, {})) - fields
            if unknown:
                raise ValueError(f'Unknown analysis.{group} settings: {sorted(unknown)}')
        gmm = analysis.get('gmm', {})
        if not 1 <= gmm.get('k_min', 1) <= gmm.get('k_max', 80):
            raise ValueError('GMM requires 1 <= k_min <= k_max')
        split = analysis.get('split', {})
        fit, evaluation = split.get('fit_fraction', .4), split.get('evaluation_fraction', .6)
        if not (0 < fit < 1 and 0 < evaluation < 1 and abs(fit + evaluation - 1) < 1e-9):
            raise ValueError('Fit/evaluation fractions must be positive and sum to one')
        if not 0 < analysis.get('monitoring', {}).get('target_far', .1) < 1:
            raise ValueError('target_far must be between zero and one')
        if not -1 <= analysis.get('alignment', {}).get('eta', .6) <= 1:
            raise ValueError('Cosine alignment eta must be between -1 and one')
        if analysis.get('smoothing', {}).get('alpha', 1) <= 0:
            raise ValueError('Laplace alpha must be positive')
    return config


def cli_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    model, collection = config.get('model', {}), config.get('collection', {})
    defaults = {key: value for key, value in collection.items() if key != 'extraction'}
    defaults.update({key: value for key, value in model.items() if key not in ('identifier', 'backend', 'chat_template_kwargs')})
    if 'identifier' in model:
        defaults['model'] = model['identifier']
    for key, value in config.get('generation', {}).items():
        defaults[{'max_new_tokens': 'max_tokens'}.get(key, key)] = value
    mapping = {
        'hidden_states_layers': 'layers', 'hidden_states_dtype': 'hidden_dtype',
        'attention_layers': 'attention_layers', 'mlp_layers': 'mlp_layers',
        'attention_save_patterns': 'attention_patterns', 'attention_save_outputs': 'attention_outputs',
        'attention_pattern_window': 'attention_window',
    }
    for key, value in config.get('capture', {}).items():
        if key in ('hidden_states_layers', 'attention_layers', 'mlp_layers'):
            value = ','.join(str(index) for index in value)
        if key == 'hidden_states':
            defaults['no_hidden_states'] = not value
        elif key == 'final_norm':
            defaults['no_final_norm'] = not value
        else:
            defaults[mapping.get(key, key)] = value
    defaults['_run_config'] = config
    return defaults
