"""Read-only full-file SHA audit and fresh answer scoring of completed GSM8K runs.

Only the audit receipt at the run root is written. Original shards/labels stay
unchanged. Existing causal-replay receipts are checked, not replayed on a GPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

from openact_core import Run
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_eval.evaluators.registry import auto_select_evaluator
from run_gsm8k_transfer import DATASET_REVISION, model_settings, prepare
from run_math_collection import atomic_json, now, sha256, verified_receipt
from upload_r2_collection import audit_shard


def audit(root, alias, prepared, config, expected=1319):
    root=Path(root); prepared=Path(prepared)
    spec=model_settings(config,alias)
    prepare(prepared)
    items=list(PreparedParquetTask(prepared).iter_items())[:expected]
    plan=json.loads((root/'job_plan.json').read_text())
    success=json.loads((root/'_SUCCESS').read_text())
    if success['samples']!=expected or plan['samples']!=expected or success['fingerprint']!=plan['fingerprint']:
        raise ValueError('Completed run coverage/fingerprint mismatch')
    if plan['model']!=spec['identifier'] or plan['model_revision']!=spec['revision'] or plan['dataset_revision']!=DATASET_REVISION:
        raise ValueError('Pinned dataset/model identity mismatch')
    shapes={'qwen2':(29,3584),'llama3':(33,4096)}
    layers,width=shapes[alias]
    def check(path):
        verified_receipt(path)  # Rehash every tensor chunk, not only the metadata.
        row=audit_shard((path,root.name,alias))
        manifest=json.loads((path/'manifest.json').read_text())
        saved=json.loads((path/'_SHARD.json').read_text())
        if saved['fingerprint']!=plan['fingerprint'] or not saved['passed']:
            raise ValueError('Shard belongs to another run')
        if not saved['verification']['fixed_shape_exact_replay_and_causality']:
            raise ValueError('Missing collector causal-replay check')
        if manifest['model']['revision']!=spec['revision']:
            raise ValueError('Model revision drift')
        if manifest['dataset']['name']!='gsm8k' or manifest['dataset']['split']!='test':
            raise ValueError('Wrong dataset/split')
        sources=manifest['custom']['dataset_sources']
        if len(sources)!=1 or sources[0]['revision']!=DATASET_REVISION or sources[0]['config']!='main':
            raise ValueError('Dataset revision/config drift')
        if manifest['prompt']['template_hash']!='507f30ec8346341e' or not manifest['prompt']['chat_template_applied']:
            raise ValueError('MATH-matched zot/chat template changed')
        if manifest['custom']['activation_extraction']!='teacher_forced_forward':
            raise ValueError('Wrong activation extraction')
        if manifest['generation']['do_sample'] or manifest['generation']['max_new_tokens']!=2048:
            raise ValueError('Decoding protocol changed')
        z=json.loads((path/'tensors.zarr/.zmetadata').read_text())['metadata']
        for group,tail in [('hidden_states',[layers,width]),('final_norm/pre',[width]),('final_norm/post',[width])]:
            for rep,first in [('per_token',row['tokens']),('prompt_last',row['samples']),('mean',row['samples'])]:
                arr=z[f'{group}/{rep}/.zarray']
                if arr['shape']!=[first,*tail] or arr['dtype']!='<f4':
                    raise ValueError(f'Activation shape/dtype mismatch: {group}/{rep}')
        labels=pq.read_table(path/'labels/correctness.parquet').to_pylist()
        evaluator=auto_select_evaluator('gsm8k')
        run=Run(path)
        finishes={}
        for sample,item,label in zip(run,items[row['start']:row['stop']],labels):
            if sample.sample_id!=item.sample_id or str(sample.ground_truth)!=str(item.ground_truth) or sample.meta['prompt_text']!=item.prompt_text:
                raise ValueError('Saved question/gold/prompt differs from prepared test slice')
            result=evaluator.evaluate_sample(sample)
            if bool(result.is_correct)!=bool(label['is_correct']) or result.extracted_answer!=label['extracted_answer']:
                raise ValueError('Independent re-scoring disagrees with saved label')
            finishes[sample.finish_reason]=finishes.get(sample.finish_reason,0)+1
        row['finish_reasons']=finishes
        return row
    started=time.monotonic()
    paths=sorted((root/alias).glob('shard_*'))
    if not paths:
        raise ValueError('No published shards')
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(check,paths))
    ids=[sid for r in rows for sid in r['sample_ids']]
    if ids!=[i.sample_id for i in items] or len(ids)!=expected:
        raise ValueError('Full ordered sample coverage failed')
    result={'passed':True,'completed_at':now(),'samples':len(ids),'shards':len(rows),
            'model':spec['identifier'],'model_revision':spec['revision'],
            'dataset_revision':DATASET_REVISION,'full_file_sha256_rechecked':True,
            'labels_independently_rescored':True,'causal_replay':'verified original per-shard checks, not repeated',
            'tokens':sum(r['tokens'] for r in rows),'correct':sum(r['correct'] for r in rows),
            'stored_bytes':sum(r['source_bytes'] for r in rows),
            'seconds':time.monotonic()-started,'plan_sha256':sha256(root/'job_plan.json'),
            'prepared_manifest_sha256':sha256(prepared/'prepared_manifest.json'),'rows':rows}
    atomic_json(root/'full_test_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',choices=['qwen2','llama3'],required=True)
    p.add_argument('--prepared',type=Path,default=Path('/lambda/nfs/dami/openact-data/prepared/gsm8k_matched_math_20260923'))
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/gsm8k_test.toml')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    result=audit(a.root,a.model,a.prepared,a.config,2 if a.smoke else 1319)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
