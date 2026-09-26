"""Two-model, three-language BELEBELE collection with a six-cell GPU smoke gate.

Reuse OpenAct's tested teacher-forced collector, activation verifier, evaluator,
and checksum-verified SSD-to-dami publisher. Never change previous runs.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from openact_collect import CollectionRunner, ModelRunner
from openact_collect.config import load_run_config, tomllib
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tasks.capability.belebele import BELEBELE_LANG_MAP
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_core import Run
from openact_core.tasks.templates import get_template
from openact_eval.evaluators.registry import auto_select_evaluator

try:
    from scripts.run_collection_matrix import SelectedPreparedTask, context_preflight, evaluate_run
    from scripts.run_math_collection import (atomic_json, sha256, now, shard_ranges,
        validate_shard, publish_with_retries, verified_receipt, check_ids, GIB)
except ModuleNotFoundError:
    from run_collection_matrix import SelectedPreparedTask, context_preflight, evaluate_run
    from run_math_collection import (atomic_json, sha256, now, shard_ranges,
        validate_shard, publish_with_retries, verified_receipt, check_ids, GIB)


ROOT = Path(__file__).resolve().parents[1]
MODELS = {'llama32': 'meta-llama/Llama-3.2-1B-Instruct', 'qwen2': 'Qwen/Qwen2-7B-Instruct'}
REVISION = '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2'


def validate_scope(cfg, generation, capture):
    if cfg['models'] != MODELS or set(cfg['model_revisions']) != set(MODELS):
        raise ValueError('Exactly the two authorized pinned models are required')
    if (cfg['languages'] != ['en', 'de', 'zh'] or cfg['samples_per_language'] != 900
            or cfg['dataset_revision'] != REVISION):
        raise ValueError('Expected pinned English, German, Chinese test, 900 rows each')
    if generation.do_sample or generation.temperature != 0 or generation.max_new_tokens != 2048:
        raise ValueError('Expected greedy generation with 2048-token cap')
    if (capture.hidden_states_layers is not None or capture.hidden_states_dtype != 'float32'
            or not all((capture.hidden_states, capture.final_norm, capture.save_per_token,
                        capture.save_mean_states, capture.save_prompt_last))):
        raise ValueError('All layers, full tokens, prompt-last, means, and both RMSNorm sides required')
    if not (cfg['smoke_samples'] == 2 and 1 <= cfg['shard_size'] <= 32 and cfg['reserve_gib'] >= 50):
        raise ValueError('Require two-row smoke and bounded shards with at least 50 GiB reserve')


def validate_item(item, lang, index):
    code = BELEBELE_LANG_MAP[lang]
    if item.sample_id != f'belebele_{code}_{index}' or item.language != lang:
        raise ValueError('Wrong language, missing row, or shuffled dataset identity')
    choices = item.meta['choices']
    if isinstance(choices, str):
        choices = json.loads(choices)
    correct = int(item.meta['correct_answer_num'])
    if len(choices) != 4 or not 1 <= correct <= 4 or item.ground_truth != 'ABCD'[correct - 1]:
        raise ValueError('BELEBELE choices and answer letter disagree')
    question = (f"{item.meta['passage']}\n\nQuestion: {item.meta['question']}\n\n"
                + '\n'.join(f'{letter}. {answer}' for letter, answer in zip('ABCD', choices)))
    expected = get_template('belebele', 'zot').format(question=question)
    if item.prompt_fields != {'question': question} or item.prompt_text != expected:
        raise ValueError('Prepared ZoT prompt differs from source passage/question/choices')


def load_data(cfg):
    data, files = {}, []
    for lang in cfg['languages']:
        path = Path(cfg['prepared_root']) / f'belebele_{lang}'
        manifest_path = path / 'prepared_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        sources = manifest['dataset_sources']
        if (manifest['task'] != 'belebele' or manifest['language'] != lang
                or manifest['split'] != 'test' or manifest['prompt_template_variant'] != 'zot'
                or len(sources) != 1 or sources[0]['name'] != 'facebook/belebele'
                or sources[0]['config'] != BELEBELE_LANG_MAP[lang]
                or sources[0]['revision'] != cfg['dataset_revision'] or sources[0]['split'] != 'test'):
            raise ValueError(f'Wrong prepared dataset version/template: {path}')
        task = PreparedParquetTask(path)
        items = list(task.iter_items())
        if len(items) != 900 or len({x.sample_id for x in items}) != 900:
            raise ValueError(f'Incomplete BELEBELE {lang} coverage')
        for i, item in enumerate(items):
            validate_item(item, lang, i)
        data[lang] = (path, items)
        files += [manifest_path, *sorted(path.glob('*.parquet'))]
    return data, files


def phase_run(mode, cfg, base, generation, capture, data, fingerprint, output, staging):
    phase = output / mode
    phase.mkdir(exist_ok=True)
    total = 12 if mode == 'smoke' else 5400
    report = dict(status='running', phase=mode, expected_samples=total, completed_samples=0,
                  pid=os.getpid(), started_at=now(), fingerprint=fingerprint, rows=[], preflight={})

    def save():
        report['updated_at'] = now()
        report['completed_samples'] = sum(r['samples'] for r in report['rows'])
        report['generated_tokens'] = sum(r['tokens'] for r in report['rows'])
        report['stored_bytes'] = sum(r['stored_bytes'] for r in report['rows'])
        atomic_json(phase / 'job_status.json', report)
        atomic_json(output / 'job_status.json', report)

    save()
    try:
        for alias, model_id in cfg['models'].items():
            runner = ModelRunner(model_id, dtype='bfloat16', device_map='cuda:0',
                                 attn_implementation='sdpa', revision=cfg['model_revisions'][alias])
            try:
                report.update(current_model=alias, activity='loading');save()
                runner.load()
                model = runner.get_model_spec()
                # Preserve historical MATH/MMLU collector behavior. Save inherited
                # generation defaults explicitly (Qwen repetition penalty is 1.05).
                resolved = runner.model.generation_config.to_dict()
                record = dict(model=model_id, revision=cfg['model_revisions'][alias],
                              model_generation_defaults=resolved, openact_overrides=asdict(generation),
                              model_layers_including_embedding=runner.probed_n_layers,
                              hidden_dim=model.hidden_dim)
                config_path = phase / f'{alias}_generation.json'
                if config_path.exists() and json.loads(config_path.read_text()) != record:
                    raise ValueError('Resolved model generation settings changed')
                atomic_json(config_path, record)
                for lang in cfg['languages']:
                    prepared, full = data[lang]
                    check = context_preflight(runner, full, generation.max_new_tokens)
                    if check['reduced_generation_budget']:
                        raise ValueError(f'Context would truncate the agreed generation budget: {check}')
                    report['preflight'][f'{alias}/{lang}'] = check
                    items = [full[0], full[-1]] if mode == 'smoke' else full
                    for start, stop in shard_ranges(len(items), cfg['shard_size']):
                        selected = items[start:stop]; ids = [i.sample_id for i in selected]
                        name = f'shard_{start:05d}_{stop:05d}'
                        dest = phase / alias / f'belebele_{lang}' / name
                        local = staging / mode / alias / f'belebele_{lang}' / name
                        identity = dict(fingerprint=fingerprint, phase=mode, model_alias=alias,
                                        model_id=model_id, language=lang, start=start, stop=stop,
                                        sample_ids=ids)
                        report.update(current_language=lang, current_shard=str(local),
                                      current_range=[start, stop], activity='checking');save()
                        if dest.exists():
                            receipt = verified_receipt(dest)
                            row = json.loads((dest / '_SHARD.json').read_text())
                            if any(row[k] != v for k, v in identity.items()):
                                raise ValueError('Published shard identity changed')
                            check_ids(Run(dest), ids)
                            row.update(run_path=str(dest), stored_bytes=receipt['stored_bytes'])
                        else:
                            if not (local / '_SHARD.json').exists():
                                if local.exists():
                                    local.rename(local.with_name(local.name + '.incomplete-' + str(time.time_ns())))
                                bound = int(1.5 * len(selected) * (generation.max_new_tokens + 3)
                                            * (runner.probed_n_layers + 2) * model.hidden_dim * 4)
                                if shutil.disk_usage(staging).free < bound + cfg['reserve_gib'] * GIB:
                                    raise OSError('Insufficient SSD for worst-case shard plus 50 GiB reserve')
                                local.parent.mkdir(parents=True, exist_ok=True)
                                resolved_run = {**base, 'model': {**base['model'], 'identifier': model_id,
                                    'revision': cfg['model_revisions'][alias], 'device_map': 'cuda:0'},
                                    'collection': {**base['collection'], 'task': 'belebele', 'language': lang,
                                    'output': str(local), 'prepared_path': str(prepared)}}
                                report['activity'] = 'collecting';save();torch.cuda.reset_peak_memory_stats()
                                stats = CollectionRunner(runner, SelectedPreparedTask(prepared, selected), local,
                                    generation_spec=generation, capture_spec=capture, run_config=resolved_run).run()
                                report['activity'] = 'activation_audit';save()
                                run = Run(local); tick = time.perf_counter()
                                checks = validate_shard(runner, run, ids)
                                for sample in run:
                                    state = sample.prompt_last_hidden_states
                                    if state.shape != (runner.probed_n_layers, model.hidden_dim) or not np.isfinite(state).all():
                                        raise ValueError('Invalid prompt-last hidden state')
                                audit_seconds = time.perf_counter() - tick
                                evaluation, seconds = evaluate_run(run, auto_select_evaluator('belebele'), 'correctness')
                                evaluation['label_path'] = 'labels/correctness.parquet'
                                row = dict(**identity, samples=len(run), tokens=stats['n_tokens_total'],
                                    collect_seconds=stats['duration_seconds'], audit_seconds=audit_seconds,
                                    eval_seconds=seconds, evaluation=evaluation, verification=checks,
                                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                    finish_reasons=[s.finish_reason for s in run], created_at=now(), passed=True)
                                atomic_json(local / '_SHARD.json', row)
                            else:
                                row = json.loads((local / '_SHARD.json').read_text())
                                if any(row[k] != v for k, v in identity.items()):
                                    raise ValueError('Staging identity changed')
                                validate_shard(runner, Run(local), ids)
                            report['activity'] = 'publishing';save()
                            row = publish_with_retries(local, dest)
                        report['rows'].append(row);save()
                        print(json.dumps(dict(phase=mode, published=f'{alias}/{lang}/{name}',
                            completed=report['completed_samples'], total=total)), flush=True)
            finally:
                runner.unload()
        for alias in cfg['models']:
            for lang in cfg['languages']:
                actual = [sid for r in report['rows'] if r['model_alias'] == alias and r['language'] == lang
                          for sid in r['sample_ids']]
                expected = data[lang][1]
                if mode == 'smoke': expected = [expected[0], expected[-1]]
                if actual != [i.sample_id for i in expected]:
                    raise ValueError(f'Missing or duplicated final coverage: {alias}/{lang}')
        assert report['completed_samples'] == total
        report.update(status='complete', activity='complete', completed_at=now());save()
        atomic_json(phase / '_SUCCESS', dict(samples=total, completed_at=now(), fingerprint=fingerprint,
                                            status_sha256=sha256(phase / 'job_status.json')))
        return report
    except BaseException:
        report.update(status='failed', traceback=traceback.format_exc());save();raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/belebele_full.toml')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    base_path = args.config.parent / cfg['base_config']; base = load_run_config(str(base_path))
    base.pop('model_catalog', None);base.pop('safety_judge', None)
    base['collection'].update(task='belebele', template='zot')
    generation = GenerationSpec(**{**base['generation'], 'max_new_tokens': cfg['max_new_tokens']})
    capture = CaptureSpec(**base['capture']);validate_scope(cfg, generation, capture)
    output, staging = Path(cfg['output_root']).resolve(), Path(cfg['local_root']).resolve()
    if output == staging or output in staging.parents or staging in output.parents:
        raise ValueError('Persistent output and SSD staging must be separate trees')
    output.mkdir(parents=True, exist_ok=True);staging.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, inputs = load_data(cfg)
        code = [Path(__file__), ROOT/'scripts/run_math_collection.py', ROOT/'scripts/run_collection_matrix.py',
                ROOT/'scripts/check_paper_collection.py', args.config, base_path,
                *sorted((ROOT/'packages').rglob('*.py')),
                *sorted((ROOT/'packages/openact-core/src/openact_core/tasks/prompts').glob('*.yaml'))]
        plan = dict(config=cfg, base=base, generation=asdict(generation), capture=asdict(capture),
                    data_sha256={str(f):sha256(f) for f in inputs},
                    code_sha256={str(f.relative_to(ROOT)):sha256(f) for f in code},
                    versions={n:version(n) for n in ('torch','transformers','numpy','zarr')},
                    expected_full=5400, expected_smoke=12,
                    extraction='Generate then teacher-force exact prompt+generated IDs, including terminal token',
                    generation_policy='Same pinned-model defaults and OpenAct overrides as original MATH/MMLU collection')
        fingerprint = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        file = output/'job_plan.json'
        if file.exists():
            if json.loads(file.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Frozen data/code/config changed; preserve run and use a new version')
        else:
            atomic_json(file,dict(**plan, fingerprint=fingerprint, frozen_at=now(),
                                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        if args.prepare_only:
            print(json.dumps(dict(prepared=True,fingerprint=fingerprint,expected_full=5400)));return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA bf16 required')
        gpu_pids = subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
        if any(pid != str(os.getpid()) for pid in gpu_pids):
            raise RuntimeError('GPU already occupied; do not overlap another collection')
        torch.set_num_threads(4)
        # Every language/model must pass genuine generation and capture before full work.
        phase_run('smoke',cfg,base,generation,capture,data,fingerprint,output,staging)
        phase_run('full',cfg,base,generation,capture,data,fingerprint,output,staging)
        atomic_json(output/'_SUCCESS',dict(samples=5400,completed_at=now(),fingerprint=fingerprint))


if __name__ == '__main__':
    main()
