import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _candidate_src_dirs(root: Path, pkg: str):
    return [
        root / pkg / 'src',
        root.parent / pkg / 'src',
        root / 'packages' / pkg / 'src',
        root.parent / 'packages' / pkg / 'src',
    ]


def _add_repo_srcs() -> None:
    for pkg in ('openact-core', 'openact-collect', 'openact-eval'):
        for src in _candidate_src_dirs(PROJECT_ROOT, pkg):
            if src.exists() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
                break


_add_repo_srcs()

MGSM_LANGS = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
BELEBELE_LANGS = ['en', 'zh', 'ja', 'de', 'fr', 'es', 'ru', 'ar', 'hi', 'ko', 'pt', 'it', 'th', 'vi', 'bn', 'sw', 'te', 'tr']
MONO_TASKS = ['gsm8k', 'mmlu', 'math', 'theoremqa', 'arc_challenge', 'commonsenseqa', 'truthfulqa', 'humaneval', 'ifeval']
SAFETY_TASKS = ['jbb', 'advbench', 'xstest']


def safety_task_kwargs(task_name: str, max_samples: int, profiles: dict) -> dict:
    kwargs = {'max_samples': max_samples, 'profiles': profiles}
    if task_name == 'jbb':
        kwargs['include_artifacts'] = False
    return kwargs
