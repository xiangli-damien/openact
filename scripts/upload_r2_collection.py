"""Audited, resumable OpenAct shard archives streamed directly from NFS to R2.

No credentials in arguments or logs. No local archive, no source deletion, no
public ACL. Every original file is SHA256 checked while packing. Every multipart
part is sent with Content-MD5, and its returned ETag is checked. The completed
object's composite ETag/size and selected range-read members are verified before
publishing a per-shard receipt. A final success marker requires every shard.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import configparser
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time
import uuid


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def read_json(path):
    """Atomic replacement on NFS can invalidate an already-open reader handle."""
    for attempt in range(8):
        try:return json.loads(Path(path).read_text())
        except OSError as exc:
            if exc.errno not in [errno.ESTALE,errno.ENOENT] or attempt==7:raise
            time.sleep(.05*(attempt+1))


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another uploader holds the lock')
        yield


def client(cfg):
    import boto3
    from botocore.config import Config
    path=Path(cfg['credentials_file']).expanduser()
    if path.stat().st_mode&0o077:raise PermissionError('Credential file must be owner-only')
    c=configparser.RawConfigParser();c.read(path);v=c['r2']
    if not v['endpoint'].startswith('https://') or not v['endpoint'].endswith('.r2.cloudflarestorage.com'):
        raise ValueError('Expected HTTPS Cloudflare R2 endpoint')
    return boto3.client('s3',endpoint_url=v['endpoint'],region_name='auto',
        aws_access_key_id=v['access_key_id'],aws_secret_access_key=v['secret_access_key'],
        config=Config(max_pool_connections=cfg['workers']*(cfg['part_workers']+2),
            retries={'max_attempts':10,'mode':'standard'},connect_timeout=15,read_timeout=180,
            request_checksum_calculation='when_required',response_checksum_validation='when_required'))


def head(s3,bucket,key):
    from botocore.exceptions import ClientError
    try:return s3.head_object(Bucket=bucket,Key=key)
    except ClientError as exc:
        if str(exc.response['Error']['Code']) in ['404','NoSuchKey','NotFound']:return None
        raise


def put_bytes(s3,cfg,key,data,content_type='application/octet-stream'):
    sha=hashlib.sha256(data).hexdigest();md5=hashlib.md5(data).hexdigest()
    existing=head(s3,cfg['bucket'],key)
    if existing:
        if existing['ContentLength']!=len(data) or existing.get('Metadata',{}).get('sha256')!=sha:
            raise ValueError(f'Refusing to overwrite a different existing object: {key}')
        if existing['ETag'].strip('"')!=md5:raise ValueError('Existing small-object ETag mismatch')
        return dict(key=key,bytes=len(data),sha256=sha,etag=md5)
    result=s3.put_object(Bucket=cfg['bucket'],Key=key,Body=data,ContentType=content_type,
        ContentMD5=base64.b64encode(bytes.fromhex(md5)).decode(),Metadata={'sha256':sha})
    if result['ETag'].strip('"')!=md5:raise ValueError('Small-object upload ETag mismatch')
    return dict(key=key,bytes=len(data),sha256=sha,etag=md5)


def audit_shard(task):
    path,run,model=task
    import pyarrow.parquet as pq
    receipt_path=path/'_COPY_VERIFIED.json';receipt=read_json(receipt_path)
    for name in ['_SHARD.json','_SUCCESS','manifest.json','data.parquet','labels/correctness.parquet','tensors.zarr/.zmetadata']:
        expected=receipt['files'][name];p=path/name
        if p.stat().st_size!=expected['bytes'] or sha256_file(p)!=expected['sha256']:raise ValueError(f'Metadata checksum mismatch: {p}')
    shard=read_json(path/'_SHARD.json');manifest=read_json(path/'manifest.json')
    expected_model={'llama32':'meta-llama/Llama-3.2-1B-Instruct','qwen2':'Qwen/Qwen2-7B-Instruct','llama3':'meta-llama/Meta-Llama-3-8B-Instruct'}[model]
    if manifest['model']['identifier']!=expected_model or shard['model_id']!=expected_model or shard['model_alias']!=model:
        raise ValueError('Wrong model under dataset/model path')
    data=pq.read_table(path/'data.parquet',columns=['sample_id','sample_idx','status','n_response_tokens']).to_pydict()
    labels=pq.read_table(path/'labels/correctness.parquet').to_pydict()
    n=shard['stop']-shard['start']
    if len(data['sample_id'])!=n or len(set(data['sample_id']))!=n:raise ValueError('Invalid shard question coverage')
    if data['sample_id']!=shard['sample_ids'] or labels['meta_sample_id']!=data['sample_id']:raise ValueError('Label/sample ID misalignment')
    if labels['sample_idx']!=data['sample_idx'] or any(x is None for x in labels['is_correct']):raise ValueError('Incomplete correctness labels')
    if any(x not in [None,''] for x in labels.get('error',[])):raise ValueError('Evaluation errors present')
    if any(labels.get('meta_gt_missing',[])):raise ValueError('Missing ground-truth labels')
    if set(data['status'])!={1}:raise ValueError('Incomplete collection rows')
    z=read_json(path/'tensors.zarr/.zmetadata')['metadata']
    for group in ['hidden_states','final_norm/pre','final_norm/post']:
        for rep in ['per_token','mean','prompt_last']:
            if f'{group}/{rep}/.zarray' not in z:raise ValueError(f'Missing full activation array: {group}/{rep}')
    tokens=sum(data['n_response_tokens'])
    if z['tokens/ids/.zarray']['shape'][0]!=tokens:raise ValueError('Token coverage mismatch')
    if z['hidden_states/per_token/.zarray']['shape'][0]!=tokens:raise ValueError('Hidden token coverage mismatch')
    for group in ['hidden_states','final_norm/pre','final_norm/post']:
        if z[f'{group}/prompt_last/.zarray']['shape'][0]!=n:raise ValueError('Prompt coverage mismatch')
    return dict(run=run,model=model,model_id=manifest['model']['identifier'],model_revision=manifest['model'].get('revision'),
        shard=path.name,relative_path=f'{run}/{model}/{path.name}',start=shard['start'],stop=shard['stop'],
        samples=n,tokens=tokens,correct=sum(labels['is_correct']),incorrect=n-sum(labels['is_correct']),
        files=len(receipt['files'])+1,source_bytes=receipt['stored_bytes']+receipt_path.stat().st_size,
        copy_receipt_sha256=sha256_file(receipt_path),sample_ids=data['sample_id'])


def prepare(cfg):
    root=Path(cfg['state_root']);root.mkdir(parents=True,exist_ok=True)
    frozen={k:v for k,v in cfg.items() if k not in ['workers','part_workers','audit_workers','retries','credentials_file']}
    if (root/'plan.json').exists():
        plan=read_json(root/'plan.json')
        if plan['config']!=frozen:raise ValueError('Upload protocol changed; use a new state directory')
        return plan
    tasks=[]
    for run in cfg['runs']:
        for model in cfg['models']:
            paths=sorted((Path(cfg['source_root'])/run/model).glob('shard_*'))
            if not paths:raise ValueError(f'No shards for {run}/{model}')
            tasks.extend((p,run,model) for p in paths)
    with ThreadPoolExecutor(cfg['audit_workers']) as pool:shards=list(pool.map(audit_shard,tasks))
    coverage=[]
    for run,target in zip(cfg['runs'],cfg['expected_counts']):
        for model in cfg['models']:
            records=[r for r in shards if r['run']==run and r['model']==model];cursor=0;ids=[]
            for r in sorted(records,key=lambda r:r['start']):
                if r['start']!=cursor:raise ValueError('Coverage gap or duplicate')
                cursor=r['stop'];ids.extend(r.pop('sample_ids'))
            if cursor!=target or len(ids)!=len(set(ids)):raise ValueError('Dataset incomplete or duplicate IDs')
            coverage.append(dict(run=run,model=model,samples=cursor,shards=len(records),
                source_bytes=sum(r['source_bytes'] for r in records),correct=sum(r['correct'] for r in records)))
    plan=dict(created_at=now(),config=frozen,coverage=coverage,shards=shards,
        source_bytes=sum(r['source_bytes'] for r in shards),files=sum(r['files'] for r in shards),
        samples=sum(r['samples'] for r in shards),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        format='One deterministic uncompressed USTAR archive per complete original OpenAct shard; standalone labels/metadata.',
        integrity='Original per-file SHA256 checked while packing; Content-MD5 and returned ETags checked per part; completed composite ETag and size checked; selected archive members range-read and SHA256 checked.',
        resume='Multipart upload ID and all accepted part ETags persist on NFS. Restart replays deterministic tar, rechecks source hashes, and skips matching R2 parts. Completed archives are immutable and skipped only after receipt verification.')
    save_json(root/'plan.json',plan)
    return plan


class HashReader:
    def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.count=0
    def read(self,n=-1):
        data=self.f.read(n);self.h.update(data);self.count+=len(data);return data


def stream_archive(source,expected,sink):
    """Return member byte offsets so R2 supports individual file range reads."""
    source=Path(source);actual=set()
    for base,dirs,files in os.walk(source):
        if any((Path(base)/d).is_symlink() for d in dirs):raise ValueError('Symlink in source archive')
        for name in files:
            p=Path(base)/name
            if p.is_symlink() or not p.is_file():raise ValueError('Nonregular source file')
            actual.add(p.relative_to(source).as_posix())
    required=set(expected)|{'_COPY_VERIFIED.json'}
    if actual!=required:raise ValueError(f'Source tree differs from verified receipt; missing={len(required-actual)}, extra={len(actual-required)}')
    index=[]
    with tarfile.open(fileobj=sink,mode='w|',format=tarfile.USTAR_FORMAT,bufsize=1024*1024,copybufsize=1024*1024) as archive:
        for name in sorted(actual):
            path=source/name;before=path.stat()
            info=tarfile.TarInfo(source.name+'/'+name);info.size=before.st_size;info.mode=0o600;info.mtime=0
            info.uid=info.gid=0;info.uname=info.gname=''
            header=info.tobuf(format=tarfile.USTAR_FORMAT);offset=archive.offset+len(header)
            with path.open('rb') as raw:
                reader=HashReader(raw);archive.addfile(info,reader)
            after=path.stat();sha=reader.h.hexdigest()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError(f'Source changed during read: {name}')
            if reader.count!=before.st_size:raise ValueError('Truncated source file')
            if name in expected and (reader.count!=expected[name]['bytes'] or sha!=expected[name]['sha256']):raise ValueError(f'Source SHA256 mismatch: {path}')
            index.append(dict(path=name,offset=offset,bytes=reader.count,sha256=sha))
    return index


class MultipartSink:
    """Bounded-memory multipart writer with durable, checksum-verified resume."""
    def __init__(self,s3,cfg,key,state_path,source_sha):
        self.s3=s3;self.cfg=cfg;self.key=key;self.path=Path(state_path);self.source_sha=source_sha
        self.size=cfg['part_mib']*1024**2;self.buffer=bytearray();self.total=0;self.sha=hashlib.sha256()
        self.next_part=1;self.pending={};self.part_md5={};self.mutex=threading.Lock()
        self.pool=ThreadPoolExecutor(cfg['part_workers']);self.completed=False
        self.state=read_json(self.path) if self.path.exists() else None
        if self.state and (self.state['key']!=key or self.state['source_sha256']!=source_sha):raise ValueError('Multipart source/key changed')
        if self.state:
            try:
                parts={}
                for page in s3.get_paginator('list_parts').paginate(Bucket=cfg['bucket'],Key=key,UploadId=self.state['upload_id']):
                    for p in page.get('Parts',[]):parts[str(p['PartNumber'])]=dict(etag=p['ETag'].strip('"'),bytes=p['Size'])
                self.state['parts']=parts
            except Exception as exc:
                if getattr(exc,'response',{}).get('Error',{}).get('Code')!='NoSuchUpload':raise
                self.state=None
        if self.state is None:
            result=s3.create_multipart_upload(Bucket=cfg['bucket'],Key=key,ContentType='application/x-tar',
                Metadata={'source-copy-receipt-sha256':source_sha,'archive-format':'ustar-v1'})
            self.state=dict(key=key,source_sha256=source_sha,upload_id=result['UploadId'],parts={},created_at=now())
        save_json(self.path,self.state)

    def _upload(self,number,data,md5):
        result=self.s3.upload_part(Bucket=self.cfg['bucket'],Key=self.key,UploadId=self.state['upload_id'],
            PartNumber=number,Body=data,ContentMD5=base64.b64encode(bytes.fromhex(md5)).decode())
        etag=result['ETag'].strip('"')
        if etag!=md5:raise ValueError('R2 part ETag differs from local MD5')
        with self.mutex:
            self.state['parts'][str(number)]=dict(etag=etag,bytes=len(data));self.state['updated_at']=now()
            save_json(self.path,self.state)
        return number

    def _collect(self,block=False):
        if not self.pending:return
        done,_=wait(self.pending,timeout=None if block else 0,return_when=FIRST_COMPLETED)
        for f in done:f.result();self.pending.pop(f)

    def _submit(self,data):
        number=self.next_part;self.next_part+=1;md5=hashlib.md5(data).hexdigest();self.part_md5[number]=md5
        old=self.state['parts'].get(str(number))
        if old:
            if old['bytes']!=len(data) or old['etag']!=md5:raise ValueError('Resumed part disagrees with deterministic source')
            return
        while len(self.pending)>=self.cfg['part_workers']:self._collect(True)
        f=self.pool.submit(self._upload,number,data,md5);self.pending[f]=number
        self._collect(False)

    def write(self,data):
        self.sha.update(data);self.total+=len(data);self.buffer.extend(data)
        while len(self.buffer)>=self.size:
            block=bytes(self.buffer[:self.size]);del self.buffer[:self.size];self._submit(block)
        return len(data)

    def finish(self):
        if self.buffer:self._submit(bytes(self.buffer));self.buffer.clear()
        while self.pending:self._collect(True)
        self.pool.shutdown(wait=True)
        expected=hashlib.md5(b''.join(bytes.fromhex(self.part_md5[n]) for n in sorted(self.part_md5))).hexdigest()+f'-{len(self.part_md5)}'
        # Persist the finalized checksums before CompleteMultipartUpload, making an uncertain completion recoverable.
        self.state.update(archive_sha256=self.sha.hexdigest(),archive_bytes=self.total,expected_etag=expected,parts_total=len(self.part_md5))
        save_json(self.path,self.state)
        result=self.s3.complete_multipart_upload(Bucket=self.cfg['bucket'],Key=self.key,UploadId=self.state['upload_id'],
            MultipartUpload={'Parts':[{'PartNumber':n,'ETag':'"'+self.part_md5[n]+'"'} for n in sorted(self.part_md5)]})
        remote=head(self.s3,self.cfg['bucket'],self.key)
        if result['ETag'].strip('"')!=expected or remote['ETag'].strip('"')!=expected or remote['ContentLength']!=self.total:
            raise ValueError('Completed object checksum or length mismatch')
        self.completed=True
        return dict(bytes=self.total,sha256=self.sha.hexdigest(),etag=expected,parts=len(self.part_md5))

    def abandon(self):
        # Keep accepted parts for the next attempt. Do not delete cloud/source data.
        self.pool.shutdown(wait=True,cancel_futures=True)


def archive_receipt_paths(cfg,rec):
    base=Path(cfg['state_root'])/'shards'/rec['relative_path']
    return base.with_suffix('.json'),base.with_suffix('.parts.json'),base.with_suffix('.index.json')


def upload_shard(s3,cfg,rec):
    started=time.monotonic();source=Path(cfg['source_root'])/rec['relative_path']
    receipt_path,parts_path,index_path=archive_receipt_paths(cfg,rec)
    key=cfg['prefix']+'/archives/'+rec['relative_path']+'.tar'
    copy_path=source/'_COPY_VERIFIED.json'
    if sha256_file(copy_path)!=rec['copy_receipt_sha256']:raise ValueError('Source receipt changed since audit')
    remote=head(s3,cfg['bucket'],key)
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if not remote or remote['ContentLength']!=receipt['archive']['bytes'] or remote['ETag'].strip('"')!=receipt['archive']['etag']:
            raise ValueError('Previously verified cloud archive missing/changed')
        return receipt
    if remote:
        # Recover a complete object if the process died between CompleteMultipartUpload and receipt publication.
        state=read_json(parts_path) if parts_path.exists() else {}
        if not index_path.exists() or remote.get('Metadata',{}).get('source-copy-receipt-sha256')!=rec['copy_receipt_sha256']:
            raise ValueError('Existing archive has no trusted local completion state; refusing overwrite')
        if state.get('archive_bytes')!=remote['ContentLength'] or state.get('expected_etag')!=remote['ETag'].strip('"'):
            raise ValueError('Uncertain completed archive failed recovery verification')
        archive=dict(bytes=state['archive_bytes'],sha256=state['archive_sha256'],etag=state['expected_etag'],parts=state['parts_total'])
        index=read_json(index_path)
    else:
        original=read_json(copy_path);sink=MultipartSink(s3,cfg,key,parts_path,rec['copy_receipt_sha256'])
        try:
            index=stream_archive(source,original['files'],sink)
            if next(r['sha256'] for r in index if r['path']=='_COPY_VERIFIED.json')!=rec['copy_receipt_sha256']:
                raise ValueError('Source receipt changed while packing')
            # All original file hashes passed before the archive can become visible.
            save_json(index_path,index)
            archive=sink.finish()
        except BaseException:
            sink.abandon();raise
    # Fetch actual remote member bytes, including labels plus representative tensor chunks.
    wanted=['labels/correctness.parquet','manifest.json','tensors.zarr/.zmetadata']
    chunks=[r for r in index if r['path'].startswith('tensors.zarr/') and not Path(r['path']).name.startswith('.') and r['bytes']>0]
    if chunks:
        wanted.extend(chunks[i]['path'] for i in sorted(set([0,len(chunks)//2,len(chunks)-1]))[:cfg['verify_range_files']])
    range_checks=[]
    for r in index:
        if r['path'] not in wanted or not r['bytes']:continue
        result=s3.get_object(Bucket=cfg['bucket'],Key=key,Range=f"bytes={r['offset']}-{r['offset']+r['bytes']-1}")
        body=result['Body'].read();result['Body'].close()
        if len(body)!=r['bytes'] or hashlib.sha256(body).hexdigest()!=r['sha256']:raise ValueError('Remote archive range verification failed')
        range_checks.append(r['path'])
    small=[]
    for name in ['data.parquet','labels/correctness.parquet','manifest.json','_SHARD.json','_SUCCESS','_COPY_VERIFIED.json','tensors.zarr/.zmetadata']:
        r=next(v for v in index if v['path']==name);data=(source/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Metadata changed before standalone upload')
        small.append(put_bytes(s3,cfg,cfg['prefix']+'/metadata/'+rec['relative_path']+'/'+name,data))
    index_obj=put_bytes(s3,cfg,cfg['prefix']+'/indices/'+rec['relative_path']+'.json',index_path.read_bytes(),'application/json')
    receipt=dict(verified_at=now(),relative_path=rec['relative_path'],key=key,bucket=cfg['bucket'],
        source_copy_receipt_sha256=rec['copy_receipt_sha256'],samples=rec['samples'],source_bytes=rec['source_bytes'],
        archive=archive,index=index_obj,standalone_metadata=small,range_checks=range_checks,
        files_verified=len(index),seconds=time.monotonic()-started)
    receipt_key=cfg['prefix']+'/receipts/'+rec['relative_path']+'.json'
    existing_receipt=head(s3,cfg['bucket'],receipt_key)
    if existing_receipt:
        response=s3.get_object(Bucket=cfg['bucket'],Key=receipt_key)
        data=response['Body'].read();response['Body'].close();old=json.loads(data)
        if old['key']!=key or old['archive']!=archive or old['source_copy_receipt_sha256']!=rec['copy_receipt_sha256']:
            raise ValueError('Existing published receipt disagrees with verified archive')
        receipt=old
    else:put_bytes(s3,cfg,receipt_key,(json.dumps(receipt,indent=2)+'\n').encode(),'application/json')
    save_json(receipt_path,receipt)
    print(json.dumps(dict(verified=rec['relative_path'],bytes=archive['bytes'],seconds=receipt['seconds'])),flush=True)
    return receipt


def run(cfg,limit=None):
    root=Path(cfg['state_root']);plan=prepare(cfg);s3=client(cfg)
    put_bytes(s3,cfg,cfg['prefix']+'/manifest.json',(root/'plan.json').read_bytes(),'application/json')
    readme=('OpenAct full MATH + MMLU collection, three models.\n'
        'archives/<run>/<model>/<shard>.tar contains the complete original shard directory.\n'
        'Extract into <run>/<model>/ with tar -xf <shard>.tar; original OpenAct layout is restored.\n'
        'metadata/ stores labels, data.parquet, manifests and source SHA256 receipts separately.\n'
        'indices/ records tar member byte offsets, sizes and SHA256; Range GET can read individual files.\n'
        'receipts/ proves source and multipart verification; _SUCCESS.json is written ONLY after every shard is verified.\n'
        'Original NFS data remains intact. No activation precision or normalization changes.\n')
    put_bytes(s3,cfg,cfg['prefix']+'/README.txt',readme.encode(),'text/plain')
    for run_name in cfg['runs']:
        for name in ['job_plan.json','job_status.json','_SUCCESS']:
            p=Path(cfg['source_root'])/run_name/name
            if p.exists():put_bytes(s3,cfg,cfg['prefix']+'/collection_metadata/'+run_name+'/'+name,p.read_bytes())
    tasks=plan['shards'][:limit] if limit else plan['shards'];started=time.monotonic();errors=[]
    def one(rec):
        for attempt in range(cfg['retries']):
            try:return upload_shard(s3,cfg,rec)
            except Exception as exc:
                # Do not expose headers, credential config or secret values in logs.
                print(json.dumps(dict(retry=rec['relative_path'],attempt=attempt+1,error_type=type(exc).__name__,message=str(exc)[:400])),flush=True)
                if isinstance(exc,(ValueError,PermissionError)) or attempt+1==cfg['retries']:raise
                time.sleep(min(60,5*2**attempt))
    def status(stage):
        verified=[];inflight_bytes=0
        for rec in plan['shards']:
            rp,pp,_=archive_receipt_paths(cfg,rec)
            if rp.exists():verified.append(read_json(rp))
            elif pp.exists():inflight_bytes+=sum(v['bytes'] for v in read_json(pp).get('parts',{}).values())
        record=dict(at=now(),pid=os.getpid(),stage=stage,verified_shards=len(verified),total_shards=len(plan['shards']),
            verified_samples=sum(r['samples'] for r in verified),total_samples=plan['samples'],
            verified_source_bytes=sum(r['source_bytes'] for r in verified),source_bytes=plan['source_bytes'],
            verified_archive_bytes=sum(r['archive']['bytes'] for r in verified),inflight_accepted_bytes=inflight_bytes,
            seconds=time.monotonic()-started,workers=cfg['workers'],part_workers=cfg['part_workers'],part_mib=cfg['part_mib'],errors=errors)
        save_json(root/'status.json',record);return record
    with ThreadPoolExecutor(cfg['workers']) as pool:
        futures={pool.submit(one,r):r for r in tasks};pending=set(futures)
        while pending:
            done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for f in done:
                try:f.result()
                except Exception as exc:errors.append(dict(shard=futures[f]['relative_path'],error_type=type(exc).__name__,message=str(exc)[:400]))
            status('uploading')
    final=status('failed' if errors else 'pilot_complete' if limit else 'verifying_completion')
    if errors:raise RuntimeError(f'{len(errors)} shards failed; preserved resumable parts and all source data')
    if not limit:
        if final['verified_shards']!=len(plan['shards']) or final['verified_samples']!=plan['samples']:raise ValueError('Incomplete final coverage')
        # Recheck all immutable archive heads before a dataset-level success marker.
        with ThreadPoolExecutor(cfg['workers']) as pool:
            for _ in pool.map(lambda r:upload_shard(s3,cfg,r),plan['shards']):pass
        final.update(stage='complete',finished_at=now(),bucket=cfg['bucket'],prefix=cfg['prefix'],plan_sha256=sha256_file(root/'plan.json'))
        success_key=cfg['prefix']+'/_SUCCESS.json'
        if head(s3,cfg['bucket'],success_key):
            response=s3.get_object(Bucket=cfg['bucket'],Key=success_key);body=response['Body'].read();response['Body'].close();old=json.loads(body)
            if old.get('plan_sha256')!=final['plan_sha256'] or old.get('verified_samples')!=plan['samples']:
                raise ValueError('Existing dataset success marker differs from verified plan')
            final=old
        else:put_bytes(s3,cfg,success_key,(json.dumps(final,indent=2)+'\n').encode(),'application/json')
        save_json(root/'_SUCCESS.json',final);save_json(root/'status.json',final)
    print(json.dumps(final),flush=True)


if __name__=='__main__':
    try:import tomllib
    except ImportError:import tomli as tomllib
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/r2_math_mmlu_upload.toml')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--workers',type=int)
    a=p.parse_args();cfg=tomllib.loads(Path(a.config).read_text())
    if a.workers:cfg['workers']=a.workers
    with lock(Path(cfg['state_root'])/'UPLOAD.lock'):
        if a.prepare_only:
            plan=prepare(cfg);print(json.dumps({k:plan[k] for k in ['coverage','source_bytes','files','samples']}),flush=True)
        else:run(cfg,a.limit)
