import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


def _candidate_src_dirs(root: Path, pkg: str):
    return [
        root / pkg / 'src',
        root.parent / pkg / 'src',
        root / 'packages' / pkg / 'src',
        root.parent / 'packages' / pkg / 'src',
    ]


for _pkg in ('openact-core', 'openact-collect', 'openact-eval'):
    for _src in _candidate_src_dirs(ROOT, _pkg):
        if _src.exists() and str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
            break

from openact_collect.inspect import inspect_task
from openact_collect.tasks.registry import TaskRegistry
from openact_core.tasks.templates import get_templates

MGSM_LANGS = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
BELEBELE_LANGS = ['en', 'zh', 'ja', 'de', 'fr', 'es', 'ru', 'ar', 'hi', 'ko', 'pt', 'it', 'th', 'vi', 'bn', 'sw', 'te', 'tr']
ALL_CONFIGS: List[Tuple[str, Dict[str, Any]]] = []
for _task_name in ['gsm8k', 'mmlu', 'math', 'theoremqa', 'arc_challenge', 'commonsenseqa', 'truthfulqa', 'humaneval', 'ifeval']:
    ALL_CONFIGS.append((_task_name, {}))
for _language in MGSM_LANGS:
    ALL_CONFIGS.append(('mgsm', {'language': _language}))
for _language in BELEBELE_LANGS:
    ALL_CONFIGS.append(('belebele', {'language': _language}))
for _task_name in ['jbb', 'advbench', 'xstest']:
    ALL_CONFIGS.append((_task_name, {'_safety': True}))


def _dataset_format_from_source(source: str, task_name: str) -> str:
    if task_name == 'mgsm':
        return 'tsv'
    if any(part in source for part in ('JailbreakBench', 'AdvBench', 'XSTest', 'xstest')):
        return 'parquet'
    return 'parquet'


def _source_url(source: str) -> Optional[str]:
    if not source or source in {'prepared_parquet', 'custom', 'unknown'}:
        return None
    return f'https://huggingface.co/datasets/{source}'


def box(title: str, width: int = 80, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n{'█' * width}")
    print(f'█ {title:<{width - 3}}█')
    print(f"{'█' * width}")


def sep(title: str = '', quiet: bool = False) -> None:
    if quiet:
        return
    if title:
        print(f"\n  ──── {title} {'─' * max(0, 60 - len(title))}")
    else:
        print(f"  {'─' * 66}")


def _ground_truth_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    ground_truths = [item.get('ground_truth') for item in items]
    n_none = sum(value is None for value in ground_truths)
    present_values = [str(value) for value in ground_truths if value is not None]
    unique_values = sorted(set(present_values))
    return {
        'total': len(ground_truths),
        'present': len(ground_truths) - n_none,
        'none': n_none,
        'unique_count': len(unique_values),
        'unique_sample': unique_values[:20] if len(unique_values) <= 20 else unique_values[:10],
    }


def _template_variants(task_name: str) -> List[Dict[str, Any]]:
    try:
        variants = get_templates(task_name)
    except Exception as exc:
        return [{'error': str(exc)}]
    out: List[Dict[str, Any]] = []
    for variant_name, template in variants.items():
        out.append({
            'name': variant_name,
            'variables': list(template.variables),
            'answer_prefix': template.answer_prefix,
            'language': template.language,
            'supports_multilingual': template.supports_multilingual,
        })
    return out


def _build_task(task_name: str, extra_kwargs: Dict[str, Any], n_items: int):
    is_safety = bool(extra_kwargs.pop('_safety', False))
    if is_safety:
        from openact_collect.schema import DEFAULT_SAFETY_PROFILES
        kwargs = {'max_samples': n_items, 'profiles': {'greedy': DEFAULT_SAFETY_PROFILES['greedy']}}
        if task_name == 'jbb':
            kwargs['include_artifacts'] = False
        kwargs.update(extra_kwargs)
        return TaskRegistry.create(task_name, **kwargs), True
    kwargs = {'max_samples': n_items}
    kwargs.update(extra_kwargs)
    return TaskRegistry.create(task_name, **kwargs), False


def inspect_one(task_name: str, extra_kwargs: Dict[str, Any], n_items: int = 3, quiet: bool = False, model_runner=None) -> Dict[str, Any]:
    lang = extra_kwargs.get('language', 'en')
    config_id = f'{task_name}_{lang}' if 'language' in extra_kwargs else task_name
    box(config_id.upper(), quiet=quiet)
    try:
        task, is_safety = _build_task(task_name, dict(extra_kwargs), n_items)
    except Exception as exc:
        if not quiet:
            print(f'\n  ❌ Failed to create task: {exc}')
        return {'config_id': config_id, 'task_name': task_name, 'language': lang, 'is_safety': bool(extra_kwargs.get('_safety')), 'error': str(exc)}
    payload = inspect_task(task, n_items=n_items, model_runner=model_runner)
    out = {
        'config_id': config_id,
        'task_name': task_name,
        'language': lang,
        'is_safety': is_safety,
        'source': task.source,
        'source_url': _source_url(task.source),
        'provenance': f'HuggingFace Hub: {task.source}' if _source_url(task.source) else task.source,
        'dataset_format': _dataset_format_from_source(task.source, task_name),
        'split': task.split,
        'data_count': payload['preview_count'],
        'estimated_total': payload.get('estimated_total'),
        'task_type': getattr(task, 'task_type', 'capability'),
        'default_template': getattr(task, 'default_template', None),
        'prompt_template': payload['prompt_template'],
        'template_variants': _template_variants(task_name),
        'task': payload['task'],
        'sample_items': payload['items'],
        'ground_truth_stats': _ground_truth_stats(payload['items']),
        'error': None,
    }
    if not quiet:
        sep('Task summary', quiet=quiet)
        print(f'  task_name:      {task.task_name}')
        print(f'  source:         {task.source}')
        print(f'  split:          {task.split}')
        print(f'  language:       {getattr(task, "language", "en")}')
        print(f'  max_samples:    {getattr(task, "max_samples", None)}')
        print(f'  task_type:      {out["task_type"]}')
        print(f'  default_tmpl:   {out["default_template"]}')
        sep('Prompt template', quiet=quiet)
        print(f'  name:           {out["prompt_template"].get("name")}')
        print(f'  variables:      {out["prompt_template"].get("variables")}')
        print(f'  answer_prefix:  {out["prompt_template"].get("answer_prefix")!r}')
        print(f'  language:       {out["prompt_template"].get("language")}')
        print(f'  requested:      {out["prompt_template"].get("requested_language")}')
        print(f'  localization:   {out["prompt_template"].get("localization_mode")}')
        sep(f'Dataset rows (first {n_items})', quiet=quiet)
        print(f'  Retrieved: {payload["preview_count"]} row(s)\n')
        for index, item in enumerate(payload['items']):
            print(f'  ┌─ item[{index}] ─────────────────────────────────────────────────┐')
            print(f'  │ sample_idx:    {item.get("sample_idx")}')
            print(f'  │ sample_id:     {item.get("sample_id")}')
            print(f'  │ ground_truth:  {item.get("ground_truth")}')
            prompt_meta = item.get('prompt_metadata', {})
            print(f'  │ prompt_lang:   {prompt_meta.get("prompt_requested_language")} ({prompt_meta.get("prompt_localization_mode")})')
            semantic_meta = item.get('semantic_meta') or {}
            if semantic_meta:
                print(f'  │ semantic:      {json.dumps(semantic_meta, ensure_ascii=False)[:120]}')
            rendered = str(item.get('rendered_prompt', ''))
            print(f'  │ rendered:      {rendered[:150]}{"…" if len(rendered) > 150 else ""}')
            model_input = item.get('model_input') or {}
            if model_input:
                chat_text = str(model_input.get('chat_text', ''))
                print(f'  │ input_tokens:  {model_input.get("n_input_tokens")}')
                print(f'  │ chat_text:     {chat_text[:150]}{"…" if len(chat_text) > 150 else ""}')
            print(f'  └──────────────────────────────────────────────────────────────┘')
        sep('Ground truth statistics', quiet=quiet)
        print(f'  total: {out["ground_truth_stats"]["total"]}, present: {out["ground_truth_stats"]["present"]}, None: {out["ground_truth_stats"]["none"]}')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect task datasets, rendered prompts, and optional model inputs')
    parser.add_argument('task', nargs='?', default=None)
    parser.add_argument('language', nargs='?', default=None)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('-o', '--out', metavar='FILE', default=None)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--model', default=None)
    parser.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    parser.add_argument('--device-map', default='auto')
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.task:
        configs = [(task_name, extra) for task_name, extra in configs if task_name == args.task]
        if args.language:
            configs = [
                (task_name, extra)
                for task_name, extra in configs
                if extra.get('language') == args.language or (not extra.get('language') and args.language == 'en')
            ]
    if not configs:
        print(f'No matching dataset: task={args.task}, language={args.language}')
        print('Available tasks: gsm8k, mgsm, mmlu, math, theoremqa, arc_challenge, commonsenseqa, belebele, truthfulqa, humaneval, ifeval, jbb, advbench, xstest')
        sys.exit(1)

    model_runner = None
    if args.model:
        from openact_collect.engine.model_manager import ModelRunner
        model_runner = ModelRunner(model_name_or_path=args.model, dtype=args.dtype, device_map=args.device_map)
        model_runner.load()

    if not args.quiet:
        print(f'\n  Inspecting {len(configs)} dataset config(s), {args.n} row(s) each')
        if args.out:
            print(f'  JSON output: {args.out}')
        if args.model:
            print(f'  Model input preview: {args.model}')
        print('  Config list:')
        for task_name, extra in configs:
            language = extra.get('language', '')
            safety = extra.get('_safety', False)
            label = f'{task_name}_{language}' if language else task_name
            if safety:
                label += ' (safety)'
            print(f'    - {label}')

    results: List[Dict[str, Any]] = []
    for task_name, extra in configs:
        try:
            results.append(inspect_one(task_name, dict(extra), n_items=args.n, quiet=args.quiet, model_runner=model_runner))
        except Exception as exc:
            results.append({'config_id': f"{task_name}_{extra.get('language', '')}" if extra.get('language') else task_name, 'task_name': task_name, 'language': extra.get('language', 'en'), 'is_safety': bool(extra.get('_safety')), 'error': str(exc)})
            if not args.quiet:
                import traceback
                print(f'\n  ❌ {task_name} raised: {exc}')
                traceback.print_exc(limit=3)

    ok = sum(result.get('error') is None for result in results)
    failed = len(results) - ok
    summary = {'ok': ok, 'failed': failed, 'total': len(results)}

    if not args.quiet:
        box('Summary')
        print(f'  OK:      {ok}')
        print(f'  Failed:  {failed}')
        print(f'  Total:   {len(results)}')
        if failed > 0:
            print('\n  ⚠ Some inspections failed — see output above.')

    if args.out:
        payload = {'configs': results, 'summary': summary, 'meta': {'n_sample_rows_per_config': args.n, 'config_count': len(configs), 'model_input_preview': args.model}}
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        if not args.quiet:
            print(f'\n  Wrote JSON to {output_path.resolve()}')


if __name__ == '__main__':
    main()
