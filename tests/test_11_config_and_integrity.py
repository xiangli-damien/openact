import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openact_collect import CollectionRunner, GenericTask
from openact_collect.cli import main, _build_task_kwargs
from openact_collect.config import load_run_config
from openact_collect.engine.offset_calculator import OffsetCalculator
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tasks.capability.arc_challenge import ARCChallengeTask
from openact_collect.tasks.capability.math import _extract_boxed_from_solution
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_core import Run
from openact_eval.evaluators.safety_evaluators import build_llamaguard_prompt, parse_llamaguard_output
from tests.test_10_runtime import make_runner

CONFIG = Path(__file__).resolve().parents[1] / 'configs/research.toml'


def test_research_config():
    config = load_run_config(str(CONFIG))
    assert config['generation']['temperature'] == 0
    assert config['capture']['final_norm']
    assert CaptureSpec(**config['capture']).hidden_states_layers is None
    assert config['collection']['task'] == 'math'
    assert config['analysis']['gmm']['k_max'] == 80
    assert config['analysis']['alignment']['eta'] == .6
    assert config['analysis']['monitoring']['target_far'] == .1


def test_capability_config_uses_full_dataset():
    config = load_run_config(str(CONFIG.with_name('capability.toml')))
    assert config['model']['identifier'] == 'Qwen/Qwen2-7B-Instruct'
    assert config['collection']['task'] == 'mmlu'
    assert 'max_samples' not in config['collection']
    assert CaptureSpec(**config['capture']).hidden_states_layers is None


def test_theoremqa_retains_image_questions_with_explicit_policy():
    from openact_collect.tasks.capability.theoremqa import TheoremQATask
    task = TheoremQATask()
    task._dataset = [
        {'Question': 'Text problem', 'Answer': '1', 'Picture': None},
        {'Question': 'Diagram problem', 'Answer': '2', 'Picture': {'bytes': b'image', 'path': None}},
    ]
    items = list(task.iter_items())
    assert task.estimate_size() == len(items) == 2
    assert [item.meta['has_image'] for item in items] == [False, True]
    assert items[1].sample_id == 'theoremqa_1'
    assert items[1].meta['image_input_policy'] == 'text_only_image_not_passed'
    assert 'Diagram problem' in task.render_prompt(items[1])
    task.max_samples = 1
    assert task.estimate_size() == len(list(task.iter_items())) == 1


def test_cli_config_and_overrides(monkeypatch, tmp_path):
    from openact_collect.engine import collector
    from openact_collect.tasks.registry import TaskRegistry
    seen = {}
    class FakeCollector:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def run(self):
            return dict(processed=2, ok=2, error=0, timeout=0, duration_seconds=1)
    monkeypatch.setattr(collector, 'CollectionRunner', FakeCollector)
    monkeypatch.setattr(TaskRegistry, 'create', lambda *args, **kwargs: GenericTask([]))
    monkeypatch.setattr('sys.argv', ['openact', 'collect', '--config', str(CONFIG), '--model', 'Qwen/Qwen2-7B-Instruct', '--max-tokens', '8', '--output', str(tmp_path)])
    main()
    assert seen['model_manager'].model_name_or_path == 'Qwen/Qwen2-7B-Instruct'
    assert seen['model_manager'].dtype_str == 'bfloat16'
    assert seen['generation_spec'].max_new_tokens == 8
    assert seen['capture_spec'].hidden_states_dtype == 'float32'
    assert seen['capture_spec'].final_norm
    assert seen['capture_spec'].hidden_states_layers is None
    assert seen['run_config']['analysis']['execution'] == 'external'


@pytest.mark.parametrize('text', [
    '[capture]\nfinal_nrom=true', '[model]\nbackend="vllm"',
    '[generation]\nmax_new_tokens=0', '[capture]\nhidden_states_dtype="bfloat16"',
    '[analysis]\nexecution="internal"', '[collection]\nmax_samples=0',
    '[capture]\nhidden_states_layers=[]',
])
def test_invalid_config_fails(tmp_path, text):
    path = tmp_path / 'bad.toml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_run_config(str(path))


@pytest.mark.parametrize('task', ['advbench', 'xstest', 'jbb'])
def test_safety_cli_constructs_without_duplicate_arguments(task):
    from openact_collect.tasks.registry import TaskRegistry
    args = SimpleNamespace(task=task, max_samples=1, profiles=None, safety_split=None)
    created = TaskRegistry.create(task, **_build_task_kwargs(args))
    assert created.name == task


@pytest.mark.parametrize('task_name', ['jbb', 'advbench', 'xstest'])
def test_safety_zot_is_applied_and_unknown_template_fails(task_name):
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.tasks.safety.base import SafetyTaskItem
    item = SafetyTaskItem(0, 'example', prompt_text='Explain safe laboratory practice.')
    raw = TaskRegistry.create(task_name, template='raw')
    cot = TaskRegistry.create(task_name, template='zot')
    assert raw.render_prompt(item) == item.prompt_text
    assert cot.render_prompt(item) == item.prompt_text + '\n\nPlease reason step by step.'
    assert cot.get_prompt_template_metadata(item)['prompt_template_hash'] != raw.get_prompt_template_metadata(item)['prompt_template_hash']
    with pytest.raises(ValueError, match='raw or zot'):
        TaskRegistry.create(task_name, template='unknown')


def test_prepared_prompts_ignore_collection_template_override(tmp_path):
    path = tmp_path / 'prepared.parquet'
    pd.DataFrame([{'prompt_text': 'Already rendered.', 'sample_id': '0'}]).to_parquet(path)
    task = PreparedParquetTask(path, template='zot', split='test')
    assert task.render_prompt(next(task.iter_items())) == 'Already rendered.'


def test_arc_numeric_choices_use_letter_labels():
    task = ARCChallengeTask()
    task._dataset = [{'question': 'Example?', 'choices': {'label': ['1', '2', '3', '4'], 'text': ['a', 'b', 'c', 'd']}, 'answerKey': '2'}]
    item = next(task.iter_items())
    assert item.ground_truth == 'B'
    assert 'B. b' in task.render_prompt(item)


def test_math_uses_last_boxed_answer():
    assert _extract_boxed_from_solution(r'An intermediate \boxed{3}, final \boxed{\frac{1}{2}}') == r'\frac{1}{2}'


def test_missing_prompt_fields_fail():
    from openact_collect.tasks.base import TaskItem
    task = ARCChallengeTask()
    with pytest.raises(KeyError):
        task.render_prompt(TaskItem(0, 'bad', prompt_fields={'question': 'missing choices'}))


def test_guard_uses_selected_tokenizer_and_strict_labels():
    class Tokenizer:
        chat_template = 'version-specific policy'
        def apply_chat_template(self, messages, tokenize):
            assert messages[-1]['role'] == 'assistant'
            return json.dumps(messages)
    assert 'assistant' in build_llamaguard_prompt('Agent', 'hello', 'hi', tokenizer=Tokenizer())
    assert parse_llamaguard_output('unsafe\nO1') == (1, 'O1')
    assert parse_llamaguard_output('not safe') == (None, None)
    with pytest.raises(ValueError):
        build_llamaguard_prompt('Agent', 'hello', 'hi')


def test_offset_fallback_repeated_text_and_partial_bytes():
    class Tokenizer:
        is_fast = False
        all_special_ids = []
        def decode(self, ids, **kwargs):
            return {1: 'a', 2: 'a a', 3: 'a a a', 4: 'a a a�', 5: 'a a a你'}[len(ids)]
    calc = OffsetCalculator(Tokenizer())
    result = calc.compute_offsets('a a a你', [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result, [[0, 1], [1, 3], [3, 5], [5, 6], [5, 6]])


def test_sentencepiece_byte_run_can_temporarily_backtrack():
    class Tokenizer:
        is_fast = False
        all_special_ids = []
        def decode(self, ids, **kwargs):
            return {1: 'a', 2: 'a 👩', 3: 'a �����', 4: 'a 👩🏽'}[len(ids)]
    result = OffsetCalculator(Tokenizer()).compute_offsets('a 👩🏽', [1, 2, 3, 4])
    np.testing.assert_array_equal(result, [[0, 1], [1, 3], [3, 4], [3, 4]])


def test_prepared_skips_null_prompts(tmp_path):
    path = tmp_path / 'part_00000.parquet'
    pd.DataFrame({'prompt_text': [None, 'hello'], 'ground_truth': [None, '2']}).to_parquet(path)
    task = PreparedParquetTask(tmp_path)
    items = list(task.iter_items())
    assert task.estimate_size() == len(items) == 1
    assert items[0].sample_idx == 0
    assert task.get_prompt_template_metadata(items[0])['prompt_template_name'] == 'prepared_raw'


def test_failed_run_has_no_success_marker(tmp_path):
    runner = make_runner()
    task = GenericTask([{'prompt': 'hello'}], template_text='{missing}')
    with pytest.raises(RuntimeError, match='successful'):
        CollectionRunner(runner, task, tmp_path, generation_spec=GenerationSpec(max_new_tokens=1)).run()
    assert not (tmp_path / '_SUCCESS').exists()


def test_mean_only_capture_and_corruption_detection(tmp_path):
    runner = make_runner()
    CollectionRunner(runner, GenericTask([{'prompt': 'hello'}]), tmp_path,
                     capture_spec=CaptureSpec(save_per_token=False),
                     generation_spec=GenerationSpec(max_new_tokens=2)).run()
    run = Run(tmp_path)
    assert run[0].get_final_norm_states('pre', 'mean').shape == (16,)
    with pytest.raises(KeyError):
        run[0].get_final_norm_states('pre')
    import zarr
    root = zarr.open_group(str(tmp_path / 'tensors.zarr'), mode='a')
    root['tokens/offsets'].resize((1, 2))
    zarr.consolidate_metadata(str(tmp_path / 'tensors.zarr'))
    assert any('offsets' in issue for issue in Run(tmp_path).validate())


def test_evaluation_pipeline_persists_and_refreshes_labels(tmp_path):
    from openact_eval import EvalPipeline
    from openact_eval.evaluators.parser_evaluator import ParserEvaluator
    runner = make_runner()
    CollectionRunner(runner, GenericTask([{'prompt': 'hello', 'answer': '2'}]), tmp_path,
                     generation_spec=GenerationSpec(max_new_tokens=2)).run()
    run = Run(tmp_path)
    assert 'is_correct' not in run.list_available_labels()
    result = EvalPipeline(run, evaluator=ParserEvaluator('gsm8k')).run_pipeline(progress=False)
    assert result['n_evaluated'] == 1
    assert (tmp_path / 'labels/correctness.parquet').exists()
    assert 'is_correct' in run.list_available_labels()
