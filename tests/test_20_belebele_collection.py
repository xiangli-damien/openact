from dataclasses import replace
from pathlib import Path

import pytest

from openact_collect.config import load_run_config, tomllib
from openact_collect.tasks.base import TaskItem
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_core.tasks.templates import get_template
from scripts.run_belebele_collection import validate_scope, validate_item


def test_full_capture_scope_and_reject_partial_capture():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root/'configs/belebele_full.toml').read_text())
    base = load_run_config(str(root/'configs/capability.toml'))
    generation, capture = GenerationSpec(**base['generation']), CaptureSpec(**base['capture'])
    validate_scope(cfg,generation,capture)
    assert len(cfg['models'])*len(cfg['languages'])*cfg['samples_per_language'] == 5400
    with pytest.raises(ValueError, match='All layers'):
        validate_scope(cfg,generation,replace(capture,save_per_token=False))
    with pytest.raises(ValueError, match='two authorized'):
        validate_scope({**cfg,'models':{'llama3':'meta-llama/Meta-Llama-3-8B-Instruct'}},generation,capture)


def test_multilingual_prompt_integrity_rejects_wrong_choice_and_language():
    question='段落\n\nQuestion: 问题\n\nA. 甲\nB. 乙\nC. 丙\nD. 丁'
    item=TaskItem(sample_idx=0,sample_id='belebele_zho_Hans_0',language='zh',ground_truth='B',
        prompt_fields={'question':question},prompt_text=get_template('belebele','zot').format(question=question),
        meta={'passage':'段落','question':'问题','choices':['甲','乙','丙','丁'],'correct_answer_num':2})
    validate_item(item,'zh',0)
    with pytest.raises(ValueError,match='answer letter'):
        validate_item(replace(item,ground_truth='A'),'zh',0)
    with pytest.raises(ValueError,match='Wrong language'):
        validate_item(item,'de',0)
    with pytest.raises(ValueError,match='ZoT prompt'):
        validate_item(replace(item,prompt_text=item.prompt_text+' changed'),'zh',0)
