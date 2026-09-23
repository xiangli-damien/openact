"""Pinned GSM8K test collection using the exact MATH zot/boxed prompt format.

Fresh study data: all 1,319 test questions, Qwen2-7B, ordinary verified OpenAct
shards including all generated-token activations and both final-norm sides.
Small local shards are copied to dami with SHA checks before staging is removed.
"""
import argparse
from dataclasses import replace, asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

from openact_collect import CollectionRunner, ModelRunner
from openact_collect.config import load_run_config
from openact_collect.data import PreparedDatasetWriter
from openact_collect.inspect import build_prepared_row
from openact_collect.schema import GenerationSpec, CaptureSpec
from openact_collect.tasks.capability.gsm8k import GSM8KTask
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_core import Run
from openact_core.tasks.templates import get_template
from openact_eval.evaluators.registry import auto_select_evaluator

from run_collection_matrix import SelectedPreparedTask, context_preflight, evaluate_run
from run_math_collection import (atomic_json, sha256, now, shard_ranges, validate_shard,
    publish_with_retries, verified_receipt, check_ids, GIB)

MODEL = 'Qwen/Qwen2-7B-Instruct'
MODEL_REVISION = 'f2826a00ceef68f0f2b946d945ecc0477ce4450c'
DATASET_REVISION = '740312add88f781978c0658806c59bc2815b9866'


class MatchedMathGSM8K(GSM8KTask):
    def get_prompt_template_for_item(self, item=None):
        return replace(get_template('math','zot'), name='gsm8k_matched_math_zot')

    def iter_items(self):
        for item in super().iter_items():
            yield replace(item, prompt_fields={'problem': item.prompt_fields['question']})


def transfer_split(question):
    # Independent of model output, outcome labels and source dataset ordering.
    key = hashlib.sha256(('gsm8k-transfer-v1\0'+' '.join(question.split())).encode()).hexdigest()
    value = int(key[:16],16)/2**64
    return 'adaptation' if value < .4 else 'validation' if value < .6 else 'confirmation'


def prepare(path):
    marker = path/'prepared_manifest.json'
    if marker.exists():
        manifest=json.loads(marker.read_text())
        if manifest['dataset_revision'] != DATASET_REVISION or sha256(path/'part_00000.parquet') != manifest['parquet_sha256']:
            raise ValueError('Prepared GSM8K identity/checksum mismatch')
        return
    if list(path.glob('*.parquet')):
        raise ValueError('Incomplete preparation exists: preserve and use a fresh path')
    task=MatchedMathGSM8K(split='test',template='matched_math_zot',dataset_revision=DATASET_REVISION)
    items=list(task.iter_items())
    if len(items)!=1319 or len({i.sample_id for i in items})!=1319 or any(i.ground_truth is None for i in items):
        raise ValueError('Expected full 1,319-row GSM8K test with numeric gold')
    writer=PreparedDatasetWriter(path,batch_rows=5000)
    counts={}
    for item in items:
        split=transfer_split(item.prompt_fields['problem'])
        counts[split]=counts.get(split,0)+1
        item=replace(item,meta={**item.meta,'transfer_split':split})
        writer.add(build_prepared_row(task,item,task.render_prompt(item)))
    writer.close()
    atomic_json(marker,{'created_at':now(),'task':'gsm8k','task_source':'openai/gsm8k',
        'task_type':'capability','split':'test','language':'en','prompt_template_variant':'matched_math_zot',
        'prompt_template':task.get_prompt_template().to_dict(),'dataset_sources':task.dataset_sources,
        'dataset_revision':DATASET_REVISION,'parquet_sha256':sha256(path/'part_00000.parquet'),
        'samples':len(items),'transfer_split_counts':counts,
        'split_rule':'sha256 normalized question; 40/20/40 adaptation/validation/confirmation, no labels'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('/lambda/nfs/dami/openact-data/prepared/gsm8k_matched_math_20260923'))
    p.add_argument('--local-root',type=Path,default=Path('/home/ubuntu/openact-gsm8k-20260923'))
    p.add_argument('--output',type=Path,default=Path('/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923'))
    p.add_argument('--shard-size',type=int,default=32)
    p.add_argument('--reserve-gib',type=int,default=50)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Two questions, separate output path required')
    args=p.parse_args()
    if args.shard_size<1 or args.reserve_gib<1:
        p.error('Positive shard size and reserve required')
    args.output.mkdir(parents=True,exist_ok=True)
    guard=(args.output/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.prepared.mkdir(parents=True,exist_ok=True);prepare(args.prepared)
    if args.prepare_only:
        print((args.prepared/'prepared_manifest.json').read_text());return
    if args.local_root.resolve()==args.output.resolve() or args.local_root.resolve() in args.output.resolve().parents or args.output.resolve() in args.local_root.resolve().parents:
        raise ValueError('Local staging and output must be separate trees')
    args.local_root.mkdir(parents=True,exist_ok=True)
    items=list(PreparedParquetTask(args.prepared).iter_items())
    if len(items)!=1319:
        raise ValueError('Prepared coverage changed')
    selected=items[:2] if args.smoke else items
    root=Path(__file__).resolve().parents[1]
    base=load_run_config(str(root/'configs/capability.toml'))
    base.pop('model_catalog',None);base.pop('safety_judge',None)
    base['model'].update(identifier=MODEL,revision=MODEL_REVISION,device_map='cuda:0')
    base['collection'].update(task='gsm8k',template='matched_math_zot',prepared_path=str(args.prepared))
    generation=GenerationSpec(**base['generation']);capture=CaptureSpec(**base['capture'])
    files=[Path(__file__),root/'scripts/run_math_collection.py',root/'scripts/run_collection_matrix.py',root/'configs/capability.toml',args.prepared/'prepared_manifest.json']
    plan={'model':MODEL,'model_revision':MODEL_REVISION,'dataset_revision':DATASET_REVISION,
          'generation':asdict(generation),'capture':asdict(capture),'samples':len(selected),
          'shard_size':args.shard_size,'mode':'smoke' if args.smoke else 'full',
          'sha256':{str(f):sha256(f) for f in files}}
    key=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    path=args.output/'job_plan.json'
    if path.exists():
        if json.loads(path.read_text())['fingerprint']!=key:
            raise ValueError('Run configuration changed; use a new directory')
    else:
        atomic_json(path,{**plan,'fingerprint':key,'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'created_at':now()})
    report={'status':'running','updated_at':now(),'completed_samples':0,'expected_samples':len(selected),
            'pid':os.getpid(),'rows':[],'fingerprint':key}
    def save():
        report['updated_at']=now();report['completed_samples']=sum(r['samples'] for r in report['rows'])
        atomic_json(args.output/'job_status.json',report)
    runner=ModelRunner(MODEL,dtype='bfloat16',device_map='cuda:0',attn_implementation='sdpa',revision=MODEL_REVISION)
    try:
        save();runner.load();report['context_preflight']=context_preflight(runner,items,generation.max_new_tokens)
        for start,stop in shard_ranges(len(selected),args.shard_size):
            name=f'shard_{start:05d}_{stop:05d}';dest=args.output/'qwen2'/name;local=args.local_root/name
            ids=[i.sample_id for i in selected[start:stop]]
            identity={'fingerprint':key,'model_alias':'qwen2','model_id':MODEL,'start':start,'stop':stop,'sample_ids':ids}
            if dest.exists():
                verified_receipt(dest);record=json.loads((dest/'_SHARD.json').read_text())
                if any(record[k]!=v for k,v in identity.items()):
                    raise ValueError('Published shard identity mismatch')
                check_ids(Run(dest),ids)
            else:
                if not (local/'_SHARD.json').exists():
                    if local.exists():
                        local.rename(local.with_name(local.name+'.failed-'+str(int(__import__('time').time()))))
                    model=runner.get_model_spec()
                    bound=int(1.5*(stop-start)*(generation.max_new_tokens+3)*(runner.probed_n_layers+2)*model.hidden_dim*4)
                    if shutil.disk_usage(args.local_root).free < bound+args.reserve_gib*GIB:
                        raise OSError('Insufficient disk for bounded shard plus reserve')
                    resolved={**base,'collection':{**base['collection'],'output':str(local)}}
                    stats=CollectionRunner(runner,SelectedPreparedTask(args.prepared,selected[start:stop]),local,
                        generation_spec=generation,capture_spec=capture,run_config=resolved).run()
                    run=Run(local);checks=validate_shard(runner,run,ids)
                    evaluation,seconds=evaluate_run(run,auto_select_evaluator('gsm8k'),'correctness')
                    evaluation['label_path']='labels/correctness.parquet'
                    record={**identity,'samples':len(run),'tokens':stats['n_tokens_total'],
                            'collect_seconds':stats['duration_seconds'],'eval_seconds':seconds,
                            'evaluation':evaluation,'verification':checks,'created_at':now(),'passed':True}
                    atomic_json(local/'_SHARD.json',record)
                else:
                    record=json.loads((local/'_SHARD.json').read_text())
                    if any(record[k]!=v for k,v in identity.items()):
                        raise ValueError('Local shard identity mismatch')
                    validate_shard(runner,Run(local),ids)
                record=publish_with_retries(local,dest)
            report['rows'].append(record);save()
            print(json.dumps({'published':name,'completed':report['completed_samples'],'total':len(selected)}),flush=True)
        if [sid for r in report['rows'] for sid in r['sample_ids']] != [i.sample_id for i in selected]:
            raise ValueError('Final full-coverage identity failure')
        report['status']='complete';save();atomic_json(args.output/'_SUCCESS',{'samples':len(selected),'fingerprint':key,'completed_at':now()})
    except BaseException:
        report.update(status='failed',traceback=traceback.format_exc());save();raise
    finally:
        runner.unload();guard.close()


if __name__=='__main__':
    main()
