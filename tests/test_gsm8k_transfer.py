import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_gsm8k_transfer import MatchedMathGSM8K, transfer_split
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
