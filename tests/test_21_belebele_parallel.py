from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts import run_belebele_parallel as parallel


@dataclass
class Settings:
    enabled: bool = True


def test_continuation_rejects_changed_config_data_code_and_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, 'ROOT', tmp_path)
    monkeypatch.setattr(parallel, 'version', lambda _: '1')
    code, source = tmp_path/'collector.py', tmp_path/'data.parquet'
    code.write_text('frozen code'); source.write_text('frozen data')
    cfg, base, settings = {'models': ['a','b']}, {'seed':42}, Settings()
    plan = dict(config=cfg, base=base, generation={'enabled':True}, capture={'enabled':True},
                data_sha256={str(source):parallel.sha256(source)},
                code_sha256={'collector.py':parallel.sha256(code)}, versions={'torch':'1'})
    verify = lambda: parallel.verify_continuation(plan,cfg,base,settings,settings,[source])
    verify()
    source.write_text('different')
    with pytest.raises(ValueError,match='source data'): verify()
    source.write_text('frozen data'); code.write_text('different')
    with pytest.raises(ValueError,match='collection code'): verify()
    code.write_text('frozen code')
    monkeypatch.setattr(parallel, 'version', lambda _: '2')
    with pytest.raises(ValueError,match='runtime'): verify()
    with pytest.raises(ValueError,match='config'):
        parallel.verify_continuation(plan,{'models':['a']},base,settings,settings,[source])


def test_coverage_requires_every_ordered_id_and_shared_disk_reserves_both():
    data={'en':(None,[SimpleNamespace(sample_id=f'id{i}') for i in range(3)])}
    row=dict(model_alias='llama32', language='en',sample_ids=['id0','id1','id2'])
    parallel.check_coverage([row],'llama32',data)
    for ids in (['id0','id1'], ['id0','id1','id1'], ['id1','id0','id2']):
        with pytest.raises(ValueError,match='coverage'):
            parallel.check_coverage([{**row,'sample_ids':ids}],'llama32',data)
    cfg=dict(shard_size=32,max_new_tokens=2048,reserve_gib=50)
    dims={'llama32':dict(model_layers_including_embedding=17,hidden_dim=2048),
          'qwen2':dict(model_layers_including_embedding=29,hidden_dim=3584)}
    together=parallel.shared_disk_bound(cfg,dims)
    single=parallel.shared_disk_bound(cfg,{'llama32':dims['llama32']})
    assert together > single > 50*parallel.GIB
