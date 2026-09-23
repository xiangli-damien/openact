"""Audit/reuse complete Qwen GSM8K; smoke, collect and audit Llama on one GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_math_collection import atomic_json, now, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/lambda/nfs/dami/openact/runs/gsm8k_test_20260923'))
    p.add_argument('--qwen-root',type=Path,default=Path('/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923'))
    p.add_argument('--local-root',type=Path,default=Path('/home/ubuntu/openact-gsm8k-llama3-20260923'))
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'.queue.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parents[1]
    state={'pid':os.getpid(),'state':'running','started_at':now(),
           'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
           'root':str(a.root),'completed_stages':[]}
    def save():
        state['updated_at']=now();atomic_json(a.root/'queue_status.json',state)
    def stage(name,script,args):
        state.update(stage=name,child_pid=None);save()
        cmd=[sys.executable,str(source/'scripts'/script),*map(str,args)]
        with (a.root/(name+'.log')).open('a') as log:
            log.write(json.dumps({'started_at':now(),'command':cmd})+'\n');log.flush()
            proc=subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=proc.pid;save()
            while proc.poll() is None:
                time.sleep(10);save()
            if proc.returncode:
                raise RuntimeError(f'{name} exited {proc.returncode}; see {name}.log')
        state['completed_stages'].append(name);save()
    def audit(name,root,model,smoke=False):
        stage(name,'audit_gsm8k_test.py',['--root',root,'--model',model,*(['--smoke'] if smoke else [])])
    try:
        save()
        audit('audit_qwen_reuse',a.qwen_root,'qwen2')
        state['stage']='waiting_for_idle_gpu';state['child_pid']=None;save()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            time.sleep(30);save()
        for mode in ['smoke','full']:
            dest=a.root/('llama3_smoke' if mode=='smoke' else 'llama3_full')
            if not (dest/'_SUCCESS').exists():
                stage('collect_llama3_'+mode,'run_gsm8k_transfer.py',[
                    '--model','llama3','--output',dest,'--local-root',a.local_root/mode,
                    '--shard-size','32','--reserve-gib','50',*(['--smoke'] if mode=='smoke' else [])])
            audit('audit_llama3_'+mode,dest,'llama3',mode=='smoke')
        rows={}
        for alias,root in [('qwen2',a.qwen_root),('llama3',a.root/'llama3_full')]:
            receipt=json.loads((root/'full_test_audit.json').read_text())
            rows[alias]={'path':str(root/alias),'samples':receipt['samples'],'correct':receipt['correct'],
                         'stored_bytes':receipt['stored_bytes'],'reused_existing':alias=='qwen2',
                         'audit_sha256':sha256(root/'full_test_audit.json')}
        atomic_json(a.root/'collection_manifest.json',{'complete':True,'completed_at':now(),
                    'dataset':'openai/gsm8k','config':'main','split':'test','models':rows})
        state.update(state='complete',stage='complete',child_pid=None);save()
    except BaseException:
        state.update(state='failed',traceback=traceback.format_exc());save();raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
