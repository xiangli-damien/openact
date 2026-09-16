"""Check parsers/matchers against every prepared capability reference answer.

This tests answer-format handling and label plumbing, not a model's accuracy or
equivalence to an external official evaluation harness.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from openact_collect.tasks.prepared import PreparedParquetTask
from openact_eval.evaluators.registry import auto_select_evaluator


def audit(path):
    task = PreparedParquetTask(path)
    evaluator = auto_select_evaluator(task.name)
    report = {'task': task.name, 'rows': 0, 'failures': []}
    for item in task.iter_items():
        if task.name == 'math':
            response = f'The final answer is \\boxed{{{item.ground_truth}}}.'
        else:
            prefix = item.meta.get('prompt_answer_prefix') or 'Answer'
            response = f'{prefix}: {item.ground_truth}'
        sample = SimpleNamespace(sample_idx=item.sample_idx, ground_truth=item.ground_truth,
                                 prompt_text=item.prompt_text, response_text=response, meta=item.meta)
        record = evaluator.evaluate_sample(sample)
        report['rows'] += 1
        if record.is_correct is not True:
            report['failures'].append({'sample_id': item.sample_id, 'answer_type': item.meta.get('answer_type'),
                                       'extracted': record.extracted_answer, 'ground_truth': item.ground_truth})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, default=Path('/lambda/nfs/dami/openact-data/prepared'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {key: audit(args.prepared_root / key) for key in
               ['math', 'mmlu', 'belebele_en', 'belebele_de', 'belebele_zh', 'theoremqa']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    for key, result in results.items():
        print(f'{key}: {result["rows"]} reference answers, {len(result["failures"])} failures')
    raise SystemExit(1 if any(result['failures'] for result in results.values()) else 0)


if __name__ == '__main__':
    main()
