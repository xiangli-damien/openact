"""Validate a prompt-only Safety Run against already published full captures.

Uses cached answers and Guard labels. This creates a separate diagnostic Run,
never regenerates answers, and never counts diagnostic duplicates toward quotas.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

try:
    from scripts import run_safety_balanced as safety
except ModuleNotFoundError:
    import run_safety_balanced as safety


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/safety_balanced_adversarial.toml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path(args.config).resolve()
    config = safety.tomllib.loads(config_path.read_text())
    root = Path(config['output'])
    base = safety.load_run_config(str(config_path.parent / config['base_config']))
    capture = safety.capture_spec_for_mode(base['capture'], 'prompt-last')
    rows = {r['sample_id']: r for p in sorted((root/'screening').glob('*/judged.json'))
            for r in json.loads(p.read_text())}
    references = {}
    for receipt in sorted(root.glob('llama2/*/*/_COPY_VERIFIED.json')):
        manifest = json.loads((receipt.parent/'manifest.json').read_text())
        if not manifest['capture']['save_per_token']:
            continue
        for i, row in pd.read_parquet(receipt.parent/'data.parquet').iterrows():
            references[row['sample_id']] = (receipt.parent, int(i))
    selected_ids = []
    for label in (True, False):
        selected_ids.append(next(sid for sid in references if rows[sid]['judge']['is_safe'] is label))
    selected_ids.append(max(references, key=lambda sid: len(rows[sid]['prompt_token_ids'])))
    selected_ids = list(dict.fromkeys(selected_ids))
    source = safety.PreparedParquetTask(config['prepared_path'])
    by_id = {item.sample_id: item for item in source.iter_items()}
    task = safety.SelectedPreparedTask(config['prepared_path'], [by_id[sid] for sid in selected_ids])
    generation = safety.GenerationSpec(**{**base['generation'], 'max_new_tokens': config['max_new_tokens']})
    runner = safety.ModelRunner(config['model'], revision=config['model_revision'],
        dtype='bfloat16', device_map='cuda:0', attn_implementation='sdpa')
    torch.set_num_threads(4)
    runner.load()
    try:
        def forbid_generation(*args, **kwargs):
            raise AssertionError('Smoke test must reuse existing answers')
        runner.model.generate = forbid_generation
        stats = safety.ReplayCollection(runner, task, output/'run', responses=rows,
            capture_spec=capture, generation_spec=generation, capture_mode='prompt-last',
            run_config={'diagnostic_only': True, 'capture': capture.to_dict()}).run()
        run = safety.Run(output/'run')
        validation = safety.validate_prompt_shard(runner, run, selected_ids)
        labels = safety.EvalResult(records=safety.label_result([rows[sid] for sid in selected_ids]),
            evaluator_name='llamaguard/Llama-Guard-3-8B',
            evaluator_config={'model_name':config['guard_model'], 'revision':config['guard_revision'],
                              'source':'cached exact screened responses'})
        labels.save_labels(output/'run', 'safety')
        labels.save_labels(output/'run', 'correctness')
        labels.save(output/'run'/'evaluation.json')
        comparisons = []
        for sample in run:
            path, index = references[sample.sample_id]
            old = safety.Run(path)[index]
            pairs = {'hidden_states': (sample.prompt_last_hidden_states, old.prompt_last_hidden_states)}
            for side in ('pre', 'post'):
                pairs[side] = (sample.get_final_norm_states(side, 'prompt_last'),
                               old.get_final_norm_states(side, 'prompt_last'))
            errors = {key: float(np.max(np.linalg.norm(a-b, axis=-1) /
                        np.maximum(np.linalg.norm(b, axis=-1), 1e-12))) for key, (a,b) in pairs.items()}
            for a, b in pairs.values():
                np.testing.assert_array_equal(a, b)
            assert sample.token_ids.tolist() == rows[sample.sample_id]['token_ids']
            comparisons.append(dict(sample_id=sample.sample_id, reference_run=str(path),
                reference_index=index, prompt_tokens=len(sample.prompt_token_ids),
                response_tokens=sample.n_tokens, relative_l2=errors))
        report = dict(passed=True, at=safety.now(), source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(), stats=stats, verification=validation,
            comparisons=comparisons, stored_bytes=sum(p.stat().st_size for p in (output/'run').rglob('*') if p.is_file()))
        safety.atomic_json(output/'summary.json', report)
        print(json.dumps(report), flush=True)
    finally:
        runner.unload()


if __name__ == '__main__':
    main()
