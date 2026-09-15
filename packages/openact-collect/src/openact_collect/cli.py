import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('openact.cli')


def parse_layers(layers_str: str) -> Optional[List[int]]:
    value = (layers_str or '').strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def _configure_logging(output_dir: str, level: int = logging.INFO) -> None:
    log_format = '%(asctime)s %(name)s %(levelname)s %(message)s'
    handlers = [logging.StreamHandler()]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(out_path / 'collect.log'))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)


def _build_task_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    task_kwargs: Dict[str, Any] = {'max_samples': getattr(args, 'max_samples', None)}
    split = getattr(args, 'split', None)
    if split is not None:
        split_text = str(split).strip()
        if split_text:
            task_kwargs['split'] = split_text
    template = getattr(args, 'template', None)
    if template:
        task_kwargs['template'] = template
    language = getattr(args, 'language', None)
    if language:
        task_kwargs['language'] = language
    prepared_path = getattr(args, 'prepared_path', None)
    if prepared_path:
        task_kwargs['prepared_path'] = prepared_path
    if getattr(args, 'dataset_revision', None):
        task_kwargs['dataset_revision'] = args.dataset_revision
    task_name = str(getattr(args, 'task', '')).lower()
    if task_name in ('jbb', 'advbench', 'xstest'):
        try:
            from openact_collect.tasks.safety.cli_extension import resolve_safety_task_kwargs
            task_kwargs.update(resolve_safety_task_kwargs(args))
            if getattr(args, '_run_config', None) and not args.profiles:
                from openact_collect.schema import GenerationProfile
                task_kwargs['profiles'] = {'configured': GenerationProfile(name='configured', max_new_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed)}
        except ImportError as exc:
            raise ImportError('Safety tasks require additional dependencies. Please ensure safety task modules are available.') from exc
    return task_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(prog='openact', description='OpenAct: LLM Activation Data Collection', formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    collect_parser = subparsers.add_parser('collect', help='Collect activation data', description='Run data collection with specified model and task.')
    collect_parser.add_argument('--config', help='Run TOML; explicit CLI flags override it')
    collect_parser.add_argument('--model', '-m', help="Model name or path (e.g. 'Qwen/Qwen2-7B-Instruct')")
    collect_parser.add_argument('--task', '-t', help="Task name (e.g. 'gsm8k', 'mmlu', 'math', 'arc_challenge', 'truthfulqa')")
    collect_parser.add_argument('--output', '-o', help='Output directory for the run')
    data_group = collect_parser.add_argument_group('data options')
    data_group.add_argument('--max-samples', '-n', type=int, default=None)
    data_group.add_argument('--split', default=None, help='Dataset split override (default: task-specific)')
    data_group.add_argument('--template', default=None)
    data_group.add_argument('--language', default=None)
    data_group.add_argument('--dataset-revision', default=None)
    data_group.add_argument('--prepared-path', default=None, help="Path to prepared parquet parts (only used when task='prepared')")
    model_group = collect_parser.add_argument_group('model options')
    model_group.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    model_group.add_argument('--device-map', default='auto')
    model_group.add_argument('--revision', default=None, help='Model and tokenizer revision')
    model_group.add_argument('--attn-implementation', choices=['eager', 'sdpa', 'flash_attention_2'], default=None)
    capture_group = collect_parser.add_argument_group('capture options')
    capture_group.add_argument('--layers', type=str, default=None)
    capture_group.add_argument('--no-hidden-states', action='store_true')
    capture_group.add_argument('--no-final-norm', action='store_true', help='Disable pre/post final normalization capture')
    capture_group.add_argument('--hidden-dtype', choices=['float16', 'float32'], default='float16')
    capture_group.add_argument('--attention', action='store_true')
    capture_group.add_argument('--attention-layers', type=str, default=None)
    capture_group.add_argument('--attention-outputs', action='store_true')
    capture_group.add_argument('--attention-patterns', action='store_true')
    capture_group.add_argument('--attention-window', type=int, default=256)
    capture_group.add_argument('--mlp', action='store_true')
    capture_group.add_argument('--mlp-layers', type=str, default=None)
    gen_group = collect_parser.add_argument_group('generation options')
    gen_group.add_argument('--max-tokens', type=int, default=2048)
    gen_group.add_argument('--temperature', type=float, default=0.0)
    gen_group.add_argument('--top-p', type=float, default=1.0)
    gen_group.add_argument('--top-k', type=int, default=None)
    gen_group.add_argument('--stop-sequences', nargs='+', default=[])
    gen_group.add_argument('--seed', type=int, default=42)
    gen_group.add_argument('--queue-size', type=int, default=8, help='Compatibility arg. The writer is synchronous; this value is ignored.')
    try:
        from openact_collect.tasks.safety.cli_extension import add_safety_arguments
        add_safety_arguments(collect_parser)
    except ImportError:
        pass

    prepare_parser = subparsers.add_parser('prepare', help='Prepare prompts/labels without running the model')
    prepare_parser.add_argument('--task', required=True)
    prepare_parser.add_argument('--output', required=True)
    prepare_parser.add_argument('--split', default=None, help='Dataset split override (default: task-specific)')
    prepare_parser.add_argument('--max-samples', type=int, default=None)
    prepare_parser.add_argument('--template', default=None)
    prepare_parser.add_argument('--language', default=None)
    prepare_parser.add_argument('--batch-rows', type=int, default=5000)
    try:
        from openact_collect.tasks.safety.cli_extension import add_safety_arguments
        add_safety_arguments(prepare_parser)
    except ImportError:
        pass

    inspect_parser = subparsers.add_parser('inspect', help='Inspect task items, rendered prompts, and optional model inputs')
    inspect_parser.add_argument('--task', required=True)
    inspect_parser.add_argument('--split', default=None)
    inspect_parser.add_argument('--max-samples', type=int, default=None)
    inspect_parser.add_argument('--template', default=None)
    inspect_parser.add_argument('--language', default=None)
    inspect_parser.add_argument('--prepared-path', default=None)
    inspect_parser.add_argument('--n', type=int, default=3)
    inspect_parser.add_argument('--output', '-o', default=None)
    inspect_parser.add_argument('--model', default=None)
    inspect_parser.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    inspect_parser.add_argument('--device-map', default='auto')
    try:
        from openact_collect.tasks.safety.cli_extension import add_safety_arguments
        add_safety_arguments(inspect_parser)
    except ImportError:
        pass

    list_parser = subparsers.add_parser('list', help='List available tasks')
    list_parser.add_argument('--verbose', '-v', action='store_true')

    info_parser = subparsers.add_parser('info', help='Show information about a run')
    info_parser.add_argument('run_dir')

    config_parser = subparsers.add_parser('config-check', help='Validate a run TOML without loading weights')
    config_parser.add_argument('path')

    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument('--config')
        config_args, _ = probe.parse_known_args()
        if config_args.config:
            from openact_collect.config import cli_defaults, load_run_config
            try:
                collect_parser.set_defaults(**cli_defaults(load_run_config(config_args.config)))
            except (ValueError, OSError) as exc:
                parser.error(str(exc))

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    if args.command == 'collect':
        for required in ('model', 'task', 'output'):
            if not getattr(args, required, None):
                parser.error(f'collect requires --{required} or its TOML setting')
        cmd_collect(args)
    elif args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'config-check':
        from openact_collect.config import load_run_config
        try:
            print(json.dumps(load_run_config(args.path), indent=2))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))


def cmd_collect(args: argparse.Namespace) -> None:
    from openact_collect.engine.collector import CollectionRunner
    from openact_collect.engine.model_manager import ModelRunner
    from openact_collect.schema import CaptureSpec, GenerationSpec
    from openact_collect.tasks.registry import TaskRegistry
    _configure_logging(args.output)
    logger.info('OpenAct Data Collection')
    logger.info('=' * 50)
    logger.info('Model: %s', args.model)
    logger.info('Task: %s', args.task)
    logger.info('Output: %s', args.output)
    if args.task.lower() == 'prepared' and not getattr(args, 'prepared_path', None):
        logger.error("--prepared-path is required when task='prepared'")
        sys.exit(2)
    task_kwargs = _build_task_kwargs(args)
    try:
        task = TaskRegistry.create(args.task, **task_kwargs)
    except ValueError as exc:
        logger.error('Task creation failed: %s', exc)
        sys.exit(1)
    logger.info('Split: %s', getattr(task, 'split', ''))
    logger.info('Loading model...')
    run_config = getattr(args, '_run_config', {})
    model_runner = ModelRunner(model_name_or_path=args.model, dtype=args.dtype, device_map=args.device_map, revision=args.revision, attn_implementation=('eager' if args.attention_patterns else args.attn_implementation), chat_template_kwargs=run_config.get('model', {}).get('chat_template_kwargs'))
    want_attention = bool(getattr(args, 'attention', False) or getattr(args, 'attention_outputs', False) or getattr(args, 'attention_patterns', False))
    want_mlp = bool(getattr(args, 'mlp', False))
    capture_spec = CaptureSpec(
        hidden_states=not bool(getattr(args, 'no_hidden_states', False)),
        final_norm=not bool(getattr(args, 'no_final_norm', False)),
        hidden_states_layers=parse_layers(getattr(args, 'layers', None) or ''),
        hidden_states_dtype=getattr(args, 'hidden_dtype', 'float16'),
        save_per_token=getattr(args, 'save_per_token', True),
        save_mean_states=getattr(args, 'save_mean_states', True),
        save_prompt_last=getattr(args, 'save_prompt_last', True),
        compute_online_metrics=getattr(args, 'compute_online_metrics', True),
        attention=want_attention,
        attention_layers=parse_layers(getattr(args, 'attention_layers', None) or ''),
        attention_save_patterns=bool(getattr(args, 'attention_patterns', False)),
        attention_save_outputs=bool(getattr(args, 'attention_outputs', False) or (want_attention and run_config.get('capture', {}).get('attention_save_outputs', True))),
        attention_pattern_window=int(getattr(args, 'attention_window', 256) or 256),
        mlp=want_mlp,
        mlp_layers=parse_layers(getattr(args, 'mlp_layers', None) or ''),
        mlp_save_output=want_mlp and getattr(args, 'mlp_save_output', True),
    )
    generation_spec = GenerationSpec(
        max_new_tokens=int(getattr(args, 'max_tokens', 2048)),
        temperature=float(getattr(args, 'temperature', 0.0)),
        top_p=float(getattr(args, 'top_p', 1.0)),
        top_k=args.top_k,
        stop_sequences=args.stop_sequences,
        do_sample=bool(getattr(args, 'do_sample', False)),
        seed=int(getattr(args, 'seed', 42)),
    )
    runner = CollectionRunner(model_manager=model_runner, task=task, output_dir=args.output, capture_spec=capture_spec, generation_spec=generation_spec, queue_size=int(getattr(args, 'queue_size', 8)), run_config=run_config)
    logger.info('Starting collection...')
    stats = runner.run()
    logger.info('Collection complete!')
    logger.info('=' * 50)
    logger.info('Total processed: %d', stats['processed'])
    logger.info('Successful: %d', stats['ok'])
    logger.info('Errors: %d', stats['error'] + stats['timeout'])
    logger.info('Duration: %.1fs', stats['duration_seconds'])
    logger.info('Output: %s', args.output)
    if stats['error'] or stats['timeout']:
        sys.exit(1)


def cmd_prepare(args: argparse.Namespace) -> None:
    from datetime import datetime
    from openact_collect.data import PreparedDatasetWriter
    from openact_collect.inspect import build_prepared_row
    from openact_collect.tasks.registry import TaskRegistry
    _configure_logging(args.output)
    logger.info('OpenAct Dataset Preparation')
    logger.info('=' * 50)
    logger.info('Task: %s', args.task)
    logger.info('Output: %s', args.output)
    task_kwargs = _build_task_kwargs(args)
    try:
        task = TaskRegistry.create(args.task, **task_kwargs)
    except ValueError as exc:
        logger.error('Task creation failed: %s', exc)
        sys.exit(1)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = PreparedDatasetWriter(out_dir, batch_rows=int(getattr(args, 'batch_rows', 5000)))
    for item in task.iter_items():
        prompt_text = task.render_prompt(item)
        if not prompt_text:
            continue
        writer.add(build_prepared_row(task, item, prompt_text))
    writer.close()
    prompt_template = task.get_prompt_template()
    prepared_manifest = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'task': task.name,
        'task_source': task.source,
        'task_type': getattr(task, 'task_type', 'capability'),
        'split': getattr(args, 'split', None) or getattr(task, 'split', ''),
        'max_samples': getattr(args, 'max_samples', None),
        'language': getattr(args, 'language', None) or getattr(task, 'language', 'en'),
        'prompt_template_variant': getattr(task, '_template_variant', None),
        'task_config': dict(getattr(task, '_config', {}) or {}),
        'prompt_template': prompt_template.to_dict(),
        'dataset_sources': task.dataset_sources,
    }
    (out_dir / 'prepared_manifest.json').write_text(json.dumps(prepared_manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info('Prepared dataset written to: %s', out_dir)


def cmd_inspect(args: argparse.Namespace) -> None:
    from openact_collect.engine.model_manager import ModelRunner
    from openact_collect.inspect import inspect_task
    from openact_collect.tasks.registry import TaskRegistry
    task_kwargs = _build_task_kwargs(args)
    try:
        task = TaskRegistry.create(args.task, **task_kwargs)
    except ValueError as exc:
        print(f'Task creation failed: {exc}', file=sys.stderr)
        sys.exit(1)
    model_runner = None
    if getattr(args, 'model', None):
        model_runner = ModelRunner(model_name_or_path=args.model, dtype=args.dtype, device_map=args.device_map)
        model_runner.load()
    payload = inspect_task(task, n_items=int(getattr(args, 'n', 3) or 3), model_runner=model_runner)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if getattr(args, 'output', None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding='utf-8')
    print(rendered)


def cmd_list(args: argparse.Namespace) -> None:
    from openact_collect.tasks.registry import TaskRegistry
    tasks = TaskRegistry.list_with_info()
    if not tasks:
        print('No tasks registered.')
        return
    print('Available tasks:')
    print('-' * 40)
    for name, info in sorted(tasks.items()):
        if args.verbose:
            print(f'\n{name}:')
            print(f"  Source: {info['source']}")
            print(f"  Language: {info['language']}")
            if info['doc']:
                doc_line = info['doc'].strip().split('\n')[0]
                print(f'  Description: {doc_line}')
        else:
            print(f'  {name}')


def cmd_info(args: argparse.Namespace) -> None:
    from openact_core import Run
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f'Error: Run directory not found: {run_dir}')
        sys.exit(1)
    try:
        run = Run(run_dir)
    except Exception as exc:
        print(f'Error loading run: {exc}')
        sys.exit(1)
    manifest = run.manifest
    print(f'Run Information: {run.run_dir.name}')
    print('=' * 50)
    print('\nModel:')
    print(f'  Name: {manifest.model.name}')
    print(f'  Layers: {manifest.model.n_layers}')
    if getattr(manifest.model, 'n_decoder_layers', None) is not None:
        print(f'  Decoder layers: {manifest.model.n_decoder_layers}')
    print(f'  Hidden dim: {manifest.model.hidden_dim}')
    print('\nDataset:')
    print(f'  Name: {manifest.dataset.name}')
    print(f'  Source: {manifest.dataset.source}')
    print(f'  Split: {manifest.dataset.split}')
    print(f'  Language: {manifest.dataset.language}')
    print('\nPrompt:')
    print(f'  Template: {manifest.prompt.template_name}')
    if getattr(manifest.prompt, 'template_variant', None):
        print(f'  Variant: {manifest.prompt.template_variant}')
    print(f'  Base language: {getattr(manifest.prompt, "language", "en")}')
    print(f'  Requested language: {getattr(manifest.prompt, "requested_language", getattr(manifest.prompt, "language", "en"))}')
    print(f'  Localization: {getattr(manifest.prompt, "localization_mode", "base")}')
    print(f'  Answer prefix: {getattr(manifest.prompt, "answer_prefix", "")!r}')
    print('\nStatistics:')
    print(f'  Total samples: {run.stats.n_samples_total}')
    print(f'  Successful: {run.stats.n_samples_ok}')
    print(f'  Errors: {run.stats.n_samples_error}')
    print(f'  Total tokens: {run.stats.n_tokens_total}')
    if run.stats.duration_seconds:
        print(f'  Duration: {run.stats.duration_seconds:.1f}s')
    print('\nCapture:')
    capture_config = manifest.capture_config or manifest.capture.to_dict()
    print(f"  Hidden states: {capture_config.get('hidden_states', False)}")
    if capture_config.get('hidden_states_layers'):
        print(f"  Hidden-state layers: {capture_config.get('hidden_states_layers')}")
    if capture_config.get('attention'):
        print(f"  Attention layers: {capture_config.get('attention_layers')}")
    if capture_config.get('mlp'):
        print(f"  MLP layers: {capture_config.get('mlp_layers')}")
    stored_columns = sorted(run._df.columns.tolist())
    print('\nStored columns:')
    print('  ' + ', '.join(stored_columns))
    labels = run.list_available_labels()
    if labels:
        print('\nAvailable labels:')
        print('  ' + ', '.join(labels))


if __name__ == '__main__':
    main()
