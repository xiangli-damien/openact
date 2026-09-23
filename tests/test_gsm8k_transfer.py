import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_gsm8k_transfer import MatchedMathGSM8K, transfer_split, model_settings
from openact_core.tasks.templates import get_template
from openact_core.tasks.parsers.numeric import NumericParser


def test_gsm_prompt_matches_math_without_changing_dataset_identity():
    task=MatchedMathGSM8K(template='matched_math_zot')
    assert task.task_name=='gsm8k' and task.source=='openai/gsm8k'
    actual=task.get_prompt_template().format(problem='There are 7 marbles; remove 2.')
    expected=get_template('math','zot').format(problem='There are 7 marbles; remove 2.')
    assert actual==expected
    assert task.get_prompt_template().name=='gsm8k_matched_math_zot'


def test_boxed_numeric_labels_and_question_based_split():
    parser=NumericParser()
    assert parser.normalize(parser.extract('Work: 2+3. Final \\boxed{5}'))=='5'
    assert parser.normalize(parser.extract('\\boxed{1,250}'))=='1250'
    assert transfer_split('a  question')==transfer_split('a question')
    assert transfer_split('a question') in {'adaptation','validation','confirmation'}


def test_both_models_use_explicit_pinned_generation_defaults():
    config=Path(__file__).resolve().parents[1]/'configs/gsm8k_test.toml'
    qwen=model_settings(config,'qwen2')
    llama=model_settings(config,'llama3')
    assert qwen['repetition_penalty']==1.05
    assert llama['repetition_penalty']==1.0
    assert qwen['revision']=='f2826a00ceef68f0f2b946d945ecc0477ce4450c'
    assert llama['identifier']=='meta-llama/Meta-Llama-3-8B-Instruct'


def test_wrong_split_rejected_before_gpu_load(tmp_path):
    import pytest
    config=Path(__file__).resolve().parents[1]/'configs/gsm8k_test.toml'
    bad=tmp_path/'bad.toml'
    bad.write_text(config.read_text().replace('split = "test"','split = "train"'))
    with pytest.raises(ValueError,match='main/test'):
        model_settings(bad,'llama3')
