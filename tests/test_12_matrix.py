from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openact_collect.tasks.safety.wildjailbreak import WildJailbreakTask
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.cli import _build_task_kwargs
from openact_eval.evaluators.parser_evaluator import ParserEvaluator
from tests.test_10_runtime import make_runner


def test_wildjailbreak_filters_exact_variant_and_never_uses_completion(monkeypatch):
    rows = [
        {'data_type': 'vanilla_benign', 'vanilla': 'benign', 'completion': 'reference'},
        {'data_type': 'adversarial_harmful', 'vanilla': 'underlying', 'adversarial': 'attack'},
        {'data_type': 'vanilla_harmful', 'vanilla': 'selected goal', 'completion': 'DO NOT LEAK'},
    ]
    task = WildJailbreakTask()
    monkeypatch.setattr(task, 'load_hf_dataset', lambda spec: rows)
    items = list(task.iter_items())
    assert len(items) == 1
    assert task.render_prompt(items[0]) == 'selected goal\n\nPlease reason step by step.'
    assert items[0].ground_truth == 'harmful'
    assert items[0].category == 'vanilla_harmful'
    assert items[0].behavior_id == 'vanilla_harmful_2'
    assert items[0].profile == 'greedy'
    assert 'DO NOT LEAK' not in str(items[0])


def test_wildjailbreak_cli_defaults_to_one_greedy_generation():
    args = SimpleNamespace(task='wildjailbreak', max_samples=None, profiles=None, safety_split=None)
    kwargs = _build_task_kwargs(args)
    assert list(kwargs['profiles']) == ['greedy']
    assert kwargs['profiles']['greedy'].n_gen == 1


@pytest.mark.parametrize('answer_type,truth,response', [
    ('bool', 'True', 'Answer: True'),
    ('list of integer', '[2, 3]', 'Answer: [2, 3]'),
    ('float', '1.5', 'Answer: 1.5'),
])
def test_explicit_theoremqa_parser_uses_row_type(answer_type, truth, response):
    sample = SimpleNamespace(meta={'answer_type': answer_type}, ground_truth=truth,
                             prompt_text='', response_text=response, sample_idx=0)
    record = ParserEvaluator('theoremqa').evaluate_sample(sample)
    assert record.is_correct is True
    assert record.meta['answer_type'] == answer_type


def test_context_budget_is_clamped_and_prompt_is_never_truncated():
    runner = make_runner()
    runner._config.max_position_embeddings = 5
    result = runner.generate(torch.tensor([[2, 3, 4]]), GenerationSpec(max_new_tokens=4), capture_spec=CaptureSpec())
    assert result.token_ids and len(result.token_ids) == 2
    assert result.finish_reason == 'context_length'
    assert result.effective_max_new_tokens == 2
    with pytest.raises(ValueError, match='no generation space'):
        runner.generate(torch.tensor([[2, 3, 4, 5, 6]]), GenerationSpec(max_new_tokens=1))


@pytest.mark.parametrize('truth,answer,correct', [
    (r'\text{Evelyn}', r'\text{Evelyn}', True),
    (r'\text{June 20}', 'June 20', True),
    (r'\text{odd}', r'\text{even}', False),
    (r'\text{(C)}', r'\text{(D)}', False),
    (r'\text{13}', '13', True),
])
def test_math_preserves_textual_answers(truth, answer, correct):
    sample = SimpleNamespace(meta={}, ground_truth=truth, prompt_text='',
                             response_text=rf'\boxed{{{answer}}}', sample_idx=0)
    result = ParserEvaluator('math').evaluate_sample(sample)
    assert result.is_correct is correct
    assert result.normalized_answer


def test_matrix_has_twenty_five_cells_and_shared_sampling():
    from openact_collect.config import tomllib
    from scripts.run_collection_matrix import select_items
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'configs/experiment_matrix.toml').read_text())
    cells = [(m, d['key']) for m in config['models'] for d in config['datasets'] if m in d.get('models', config['models'])]
    assert len(cells) == 25
    assert [cell for cell in cells if cell[1] == 'wildjailbreak'] == [('llama2', 'wildjailbreak')]
    a = select_items(list(range(100)), 3, 42, 'math')
    assert a == select_items(list(range(100)), 3, 42, 'math')
    assert len(set(a)) == 3


def test_cost_estimate_scales_each_cell_without_double_counting():
    from scripts.estimate_matrix_cost import estimate
    config = {'models': {'tiny': 'tiny'}, 'datasets': [{'key': 'math', 'expected_samples': 100}], 'max_new_tokens': 2048}
    row = {'model_alias': 'tiny', 'dataset': 'math', 'samples': 2, 'tokens': 20,
           'collect_seconds': 10, 'eval_seconds': 2, 'stored_bytes': 1000,
           'response_tokens': [10, 10], 'finish_reasons': ['eos', 'eos'], 'passed': True}
    result = estimate(config, [{'rows': [row]}, {'rows': [row]}])
    assert len(result['rows']) == 1
    assert result['totals']['total_hours'] == pytest.approx(600 / 3600)
    assert result['totals']['stored_tb'] == pytest.approx(50000 / 1e12)
    assert result['missing_cells'] == []
