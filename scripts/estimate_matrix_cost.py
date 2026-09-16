"""Extrapolate measured collection/evaluation time and compressed bytes from pilots.

These are small-sample planning estimates, not guarantees or confidence intervals.
Token-length scenarios show the cost if every response has the specified length.
"""
import argparse
import json
from pathlib import Path

from openact_collect.config import tomllib


def estimate(config, reports):
    measured = {}
    for report in sorted(reports, key=lambda r: r.get('created_at', '')):
        for row in report['rows']:
            if row.get('collect_seconds') and row.get('tokens') and row.get('stored_bytes'):
                measured[(row['model_alias'], row['dataset'])] = row
    rows, missing = [], []
    for alias in config['models']:
        for dataset in config['datasets']:
            if alias not in dataset.get('models', config['models']):
                continue
            key = (alias, dataset['key'])
            pilot = measured.get(key)
            if not pilot:
                missing.append('/'.join(key))
                continue
            n = pilot['samples']
            full = dataset['expected_samples']
            factor = full / n
            token_count = pilot['tokens']
            sec_per_token = pilot['collect_seconds'] / token_count
            bytes_per_token = pilot['stored_bytes'] / token_count
            verification = pilot.get('verification', {})
            # Stored-array scans cover all rows; the deep replay checks one
            # representative sample per cell and is paid only once.
            verification_hours = (verification.get('stored_array_check_seconds', 0) * factor
                                  + verification.get('causal_replay_seconds', 0)) / 3600
            scenarios = []
            for length in (128, 256, 512, 1024, config['max_new_tokens']):
                scenarios.append({'response_tokens': length,
                                  'collection_hours': full * length * sec_per_token / 3600,
                                  'stored_gb': full * length * bytes_per_token / 1e9})
            rows.append({'model': alias, 'dataset': dataset['key'], 'full_samples': full,
                         'pilot_samples': n, 'pilot_response_tokens': pilot['response_tokens'],
                         'mean_response_tokens': token_count / n,
                         'pilot_limit_hits': sum(r in ('length', 'context_length') for r in pilot['finish_reasons']),
                         'collection_hours': pilot['collect_seconds'] * factor / 3600,
                         'evaluation_hours': pilot.get('eval_seconds', 0) * factor / 3600,
                         'verification_hours': verification_hours,
                         'stored_gb': pilot['stored_bytes'] * factor / 1e9,
                         'eval_verified': pilot.get('passed', False), 'scenarios': scenarios})
    total_collect = sum(r['collection_hours'] for r in rows)
    total_eval = sum(r['evaluation_hours'] for r in rows)
    total_verify = sum(r['verification_hours'] for r in rows)
    total_gb = sum(r['stored_gb'] for r in rows)
    return {'method': 'Per-cell observed seconds/sample and bytes/sample times full row count',
            'scope': 'One GPU, sequential models, all layers, float32 raw storage, both final norm sides',
            'caveats': ['Three-sample cells are a coarse pilot, not a benchmark-quality timing study.',
                        'Length-capped samples do not establish natural completion lengths.',
                        'Token scenarios hold measured seconds/byte cost per token constant; prefill and IO vary.',
                        'Includes stored-array scans and one deep replay per cell; excludes downloads, model loading and retries.',
                        'Cached pilot reads and provider storage performance may differ during a large run.',
                        'GB/TB are decimal; filesystem quotas and provider charges are separate.'],
            'rows': rows, 'missing_cells': missing,
            'totals': {'full_samples': sum(r['full_samples'] for r in rows),
                       'collection_hours': total_collect, 'evaluation_hours': total_eval,
                       'verification_hours': total_verify,
                       'total_hours': total_collect + total_eval + total_verify, 'stored_tb': total_gb / 1000,
                       'planning_hours_with_50pct_headroom': 1.5 * (total_collect + total_eval + total_verify),
                       'planning_tb_with_50pct_headroom': 1.5 * total_gb / 1000},
            'scenario_totals': [
                {'response_tokens': length,
                 'collection_hours': sum(next(s['collection_hours'] for s in r['scenarios'] if s['response_tokens'] == length) for r in rows),
                 'stored_tb': sum(next(s['stored_gb'] for s in r['scenarios'] if s['response_tokens'] == length) for r in rows) / 1000}
                for length in (128, 256, 512, 1024, config['max_new_tokens'])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/experiment_matrix.toml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    result = estimate(config, [json.loads(p.read_text()) for p in args.reports])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'totals': result['totals'], 'missing_cells': result['missing_cells']}, indent=2))


if __name__ == '__main__':
    main()
