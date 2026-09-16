"""Recheck existing matrix artifacts and refresh labels without generating again.

The original pilot report is preserved. Collection timings are read from each
run's manifest when an earlier verification failure omitted them from the report.
"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

from openact_collect import ModelRunner
from openact_core import Run
from openact_eval.evaluators.registry import auto_select_evaluator
from openact_eval.evaluators.safety_evaluators import LlamaGuardEvaluator

try:
    from scripts.run_collection_matrix import directory_bytes, evaluate_run, verify_run
except ModuleNotFoundError:
    from run_collection_matrix import directory_bytes, evaluate_run, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        parser.error('Preserve the original report by choosing a new output')
    report = json.loads(args.report.read_text())
    report['original_errors'] = report['errors']
    report.update(errors=[], passed=False, rechecked_at=datetime.now(timezone.utc).isoformat())
    config = report['config']
    datasets = {d['key']: d for d in config['datasets']}
    pending = []
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for alias in report['models_requested']:
        runner = ModelRunner(config['models'][alias], dtype='bfloat16', device_map='cuda:0',
                             attn_implementation='sdpa', revision=config['model_revisions'][alias])
        try:
            runner.load()
            for row in [r for r in report['rows'] if r['model_alias'] == alias]:
                row['passed'] = False
                row.pop('error', None)
                try:
                    path = args.report.parent / alias / row['dataset']
                    run = Run(path)
                    row.update(run_path=str(path), samples=len(run), model_revision=run.manifest.model.revision,
                               collect_seconds=run.manifest.stats.duration_seconds,
                               tokens=sum(s.n_tokens for s in run), response_tokens=[s.n_tokens for s in run],
                               finish_reasons=[s.finish_reason for s in run], stored_bytes=directory_bytes(path))
                    row['verification'] = verify_run(runner, run)
                    if datasets[row['dataset']].get('evaluator') == 'llamaguard':
                        pending.append((row, path))
                    else:
                        summary, seconds = evaluate_run(run, auto_select_evaluator(datasets[row['dataset']]['task']),
                                                        'correctness')
                        row.update(evaluation=summary, eval_seconds=seconds, passed=True)
                    print(f'{alias}/{row["dataset"]}: exact replay/causality passed; evaluated={row["passed"]}', flush=True)
                except Exception as exc:
                    row['error'] = str(exc)
                    report['errors'].append({'model': alias, 'dataset': row['dataset'], 'error': str(exc)})
                    print(f'FAILED {alias}/{row["dataset"]}: {exc}', flush=True)
                finally:
                    save()
        finally:
            runner.unload()
            gc.collect()
    for row, path in pending:
        try:
            judge = LlamaGuardEvaluator(model_name=config['guard_model'], revision=config['guard_revision'],
                                        dtype='bfloat16', device_map='cuda:0')
            summary, seconds = evaluate_run(Run(path), judge, 'safety')
            row.update(evaluation=summary, eval_seconds=seconds, guard_model=config['guard_model'],
                       guard_revision=judge.revision, stored_bytes=directory_bytes(path), passed=True)
            print('llama2/wildjailbreak: Guard evaluation passed', flush=True)
        except Exception as exc:
            row['error'] = str(exc)
            report['errors'].append({'model': row['model_alias'], 'dataset': row['dataset'], 'error': str(exc)})
        finally:
            save()
    report['passed'] = (len(report['rows']) == report['expected_cells']
                        and all(r['passed'] for r in report['rows']) and not report['errors'])
    save()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
