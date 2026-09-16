"""Collect/evaluate the selected matrix and measure pilot costs on the target disk.

Default: three deterministic examples per cell, with the full generation budget.
Use --full only to launch the entire dataset matrix. Existing run paths are never
overwritten; use a fresh --output for a new pilot.
"""
import argparse
import gc
import hashlib
import json
import random
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from openact_collect import CollectionRunner, ModelRunner
from openact_collect.config import load_run_config, tomllib
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_core import Run
from openact_eval import EvalPipeline
from openact_eval.evaluators.registry import auto_select_evaluator
from openact_eval.evaluators.safety_evaluators import LlamaGuardEvaluator

try:
    from scripts.check_paper_collection import verify_saved_sample
except ModuleNotFoundError:
    from check_paper_collection import verify_saved_sample


class SelectedPreparedTask(PreparedParquetTask):
    def __init__(self, path, items):
        super().__init__(path)
        self.selected = [replace(item, sample_idx=i) for i, item in enumerate(items)]

    def estimate_size(self):
        return len(self.selected)

    def iter_items(self):
        yield from self.selected


def select_items(items, n, seed, key):
    if n >= len(items):
        return items
    derived = seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    indices = sorted(random.Random(derived).sample(range(len(items)), n))
    return [items[i] for i in indices]


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def context_preflight(runner, items, generation_budget):
    lengths = []
    for start in range(0, len(items), 512):
        texts = [runner.render_chat_text([{'role': 'user', 'content': item.prompt_text}])
                 for item in items[start:start + 512]]
        encoded = runner.tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
        lengths.extend(map(len, encoded))
    limit = getattr(runner.config, 'max_position_embeddings', None)
    result = {'max_prompt_tokens': max(lengths), 'context_limit': limit,
              'no_response_space': sum(n >= limit for n in lengths) if limit else 0,
              'reduced_generation_budget': sum(n + generation_budget > limit for n in lengths) if limit else 0}
    if result['no_response_space']:
        raise ValueError(f'{result["no_response_space"]} full-dataset prompts exceed model context: {result}')
    return result


def verify_run(runner, run):
    errors = run.validate()
    if errors or not run.is_complete or run.n_valid != len(run):
        raise AssertionError(f'Invalid/incomplete collection: {errors}')
    width = runner.get_model_spec().hidden_dim
    for sample in run:
        states = sample.hidden_states
        if states.shape != (sample.n_tokens, runner.probed_n_layers, width):
            raise AssertionError('Missing token/layer activations')
        if not np.isfinite(states).all():
            raise AssertionError('Nonfinite activations')
        np.testing.assert_array_equal(states[:, -1], sample.get_final_norm_states('post'))
        np.testing.assert_allclose(states.mean(0), sample.mean_hidden_states, rtol=1e-5, atol=1e-6)
    first = run[0]
    return verify_saved_sample(runner, first, prefix_lengths=[0, 1, min(8, first.n_tokens), first.n_tokens],
                               fixed_shape=True)


def evaluate_run(run, evaluator, label):
    started = time.perf_counter()
    pipeline = EvalPipeline(run, evaluator=evaluator, label_name=label)
    summary = pipeline.run_pipeline(progress=False)
    if summary['n_error'] or summary['n_evaluated'] != len(run):
        raise AssertionError(f'Incomplete evaluation: {summary}')
    saved = Path(summary['label_path'])
    if not saved.exists():
        raise AssertionError('Evaluation did not persist labels')
    summary['pipeline_seconds'] = time.perf_counter() - started
    summary['evaluation_seconds'] = pipeline.result.wall_time_s
    summary['setup_teardown_and_io_seconds'] = summary['pipeline_seconds'] - pipeline.result.wall_time_s
    # Do not extrapolate one-time judge loading/download overhead per sample.
    return summary, pipeline.result.wall_time_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/experiment_matrix.toml'))
    parser.add_argument('--models', nargs='+', help='Model aliases; default: entire matrix')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--full', action='store_true', help='Collect every row instead of a smoke pilot')
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('CUDA with bf16 is required')
    path = Path(args.config)
    config = tomllib.loads(path.read_text())
    base = load_run_config(str(path.parent / config['base_config']))
    aliases = args.models or list(config['models'])
    if set(aliases) - set(config['models']):
        parser.error('Unknown model alias')
    n = config['smoke_samples'] if args.samples is None else args.samples
    if n < 1:
        parser.error('--samples must be positive')
    budget = config['max_new_tokens'] if args.max_tokens is None else args.max_tokens
    generation = GenerationSpec(**{**base['generation'], 'max_new_tokens': budget})
    if generation.temperature != 0 or generation.do_sample:
        parser.error('This matrix requires greedy decoding')
    capture = CaptureSpec(**base['capture'])
    if capture.hidden_states_layers is not None or capture.hidden_states_dtype != 'float32':
        parser.error('This matrix requires all hidden layers and float32 storage')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'mode': 'full' if args.full else 'smoke',
              'config': config, 'models_requested': aliases, 'samples_per_cell': n,
              'max_new_tokens': budget, 'gpu': torch.cuda.get_device_name(), 'rows': [], 'errors': [], 'passed': False}
    def save():
        (args.output / 'matrix_report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    loaded_data = {}
    pending_safety = []
    for alias in aliases:
        model_id = config['models'][alias]
        runner = ModelRunner(model_id, dtype='bfloat16', device_map='cuda:0', attn_implementation='sdpa',
                             revision=config.get('model_revisions', {}).get(alias))
        load_started = time.perf_counter()
        try:
            runner.load()
            load_seconds = time.perf_counter() - load_started
            for dataset in config['datasets']:
                if alias not in dataset.get('models', aliases):
                    continue
                key = dataset['key']
                row = {'model_alias': alias, 'model_id': model_id, 'dataset': key,
                       'expected_samples': dataset['expected_samples'], 'passed': False}
                report['rows'].append(row)
                try:
                    prepared = Path(config['prepared_root']) / key
                    if key not in loaded_data:
                        task = PreparedParquetTask(prepared)
                        if task.get_prompt_template_metadata()['prompt_template_variant'] != 'zot':
                            raise ValueError(f'{key} is not prepared with zot')
                        loaded_data[key] = list(task.iter_items())
                    all_items = loaded_data[key]
                    if len(all_items) != dataset['expected_samples']:
                        raise ValueError(f'{key}: expected {dataset["expected_samples"]} rows, got {len(all_items)}')
                    row['context_preflight'] = context_preflight(runner, all_items, budget)
                    selected = all_items if args.full else select_items(all_items, n, config['seed'], key)
                    row['sample_ids'] = [item.sample_id for item in selected]
                    task = SelectedPreparedTask(prepared, selected)
                    run_path = args.output / alias / key
                    torch.cuda.reset_peak_memory_stats()
                    stats = CollectionRunner(runner, task, run_path, capture_spec=capture,
                                             generation_spec=generation,
                                             run_config={**base, 'matrix': config, 'matrix_dataset': key}).run()
                    run = Run(run_path)
                    row.update(run_path=str(run_path), samples=len(run), model_revision=run.manifest.model.revision,
                               model_load_seconds=load_seconds, collect_seconds=stats['duration_seconds'],
                               tokens=stats['n_tokens_total'],
                               response_tokens=[s.n_tokens for s in run],
                               finish_reasons=[s.finish_reason for s in run],
                               stored_bytes=directory_bytes(run_path),
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    row['verification'] = verify_run(runner, run)
                    if dataset.get('evaluator') == 'llamaguard':
                        pending_safety.append((row, run_path))
                    else:
                        summary, seconds = evaluate_run(run, auto_select_evaluator(dataset['task']), 'correctness')
                        row.update(evaluation=summary, eval_seconds=seconds, passed=True)
                    row['stored_bytes'] = directory_bytes(run_path)
                    print(f'{alias}/{key}: collected {len(run)} samples, {row["tokens"]} tokens; evaluated={row["passed"]}', flush=True)
                except Exception as exc:
                    row['error'] = str(exc)
                    report['errors'].append({'model': alias, 'dataset': key, 'error': str(exc)})
                    print(f'FAILED {alias}/{key}: {exc}', flush=True)
                finally:
                    save()
        except Exception as exc:
            report['errors'].append({'model': alias, 'stage': 'model_load', 'error': str(exc)})
            print(f'FAILED loading {alias}: {exc}', flush=True)
        finally:
            runner.unload()
            gc.collect()
            save()
    # Generation models have been unloaded before loading the safety judge.
    for row, run_path in pending_safety:
        try:
            judge = LlamaGuardEvaluator(model_name=config['guard_model'], dtype='bfloat16', device_map='cuda:0',
                                        revision=config.get('guard_revision'))
            summary, seconds = evaluate_run(Run(run_path), judge, 'safety')
            row.update(evaluation=summary, eval_seconds=seconds, guard_model=config['guard_model'],
                       guard_revision=judge.revision, stored_bytes=directory_bytes(run_path), passed=True)
        except Exception as exc:
            row['error'] = str(exc)
            report['errors'].append({'model': row['model_alias'], 'dataset': row['dataset'], 'stage': 'guard', 'error': str(exc)})
        finally:
            save()
    expected_cells = sum(alias in d.get('models', aliases) for alias in aliases for d in config['datasets'])
    report['expected_cells'] = expected_cells
    report['passed'] = len(report['rows']) == expected_cells and all(r['passed'] for r in report['rows']) and not report['errors']
    save()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
