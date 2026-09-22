import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upload_r2_collection as upload


def test_nfs_atomic_status_replacement_retries_stale_handle(tmp_path,monkeypatch):
    import errno
    p=tmp_path/'status.json';p.write_text('{"verified": 1}')
    original=Path.read_text;calls=[]
    def flaky(path,*args,**kwargs):
        calls.append(path)
        if len(calls)==1:raise OSError(errno.ESTALE,'Stale file handle')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',flaky)
    monkeypatch.setattr(upload.time,'sleep',lambda _:None)
    assert upload.read_json(p)=={'verified':1} and len(calls)==2


def source_fixture(tmp_path):
    source=tmp_path/'shard_00000_00002';(source/'labels').mkdir(parents=True)
    files={'data.parquet':b'question answer'*35,'labels/correctness.parquet':b'true false',
           'tensors.zarr/hidden_states/per_token/0.0.0':bytes(range(256))*40}
    expected={}
    for name,data in files.items():
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        expected[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    (source/'_COPY_VERIFIED.json').write_text(json.dumps(dict(files=expected)))
    return source,expected


def test_archive_is_complete_deterministic_and_range_readable(tmp_path):
    source,expected=source_fixture(tmp_path);a=io.BytesIO()
    index=upload.stream_archive(source,expected,a);data=a.getvalue()
    b=io.BytesIO();second=upload.stream_archive(source,expected,b)
    assert data==b.getvalue() and index==second
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        assert len(tar.getmembers())==len(expected)+1
        for item in index:
            raw=(source/item['path']).read_bytes()
            assert tar.extractfile(source.name+'/'+item['path']).read()==raw
            assert data[item['offset']:item['offset']+item['bytes']]==raw


def test_corrupt_missing_or_extra_source_rejected(tmp_path):
    source,expected=source_fixture(tmp_path)
    (source/'data.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA256'):upload.stream_archive(source,expected,io.BytesIO())
    (source/'data.parquet').unlink()
    with pytest.raises(ValueError,match='differs'):upload.stream_archive(source,expected,io.BytesIO())


class FakeS3:
    def __init__(self):self.parts={};self.calls=[];self.object=None;self.meta={}
    def create_multipart_upload(self,**kwargs):self.meta=kwargs['Metadata'];return {'UploadId':'test'}
    def upload_part(self,**kwargs):
        number=kwargs['PartNumber'];data=kwargs['Body'];md5=hashlib.md5(data).hexdigest()
        assert kwargs['ContentMD5']==base64.b64encode(bytes.fromhex(md5)).decode()
        self.parts[number]=data;self.calls.append(number);return {'ETag':'"'+md5+'"'}
    def get_paginator(self,name):
        assert name=='list_parts';return self
    def paginate(self,**kwargs):
        return [{'Parts':[dict(PartNumber=n,Size=len(d),ETag='"'+hashlib.md5(d).hexdigest()+'"') for n,d in self.parts.items()]}]
    def complete_multipart_upload(self,**kwargs):
        parts=kwargs['MultipartUpload']['Parts'];ids=[r['PartNumber'] for r in parts]
        assert ids==sorted(self.parts)
        self.object=b''.join(self.parts[n] for n in ids)
        etag=hashlib.md5(b''.join(hashlib.md5(self.parts[n]).digest() for n in ids)).hexdigest()+f'-{len(ids)}'
        self.etag=etag;return {'ETag':'"'+etag+'"'}


def test_multipart_verified_resume_and_bounded_chunks(tmp_path,monkeypatch):
    s3=FakeS3();cfg=dict(bucket='bucket',part_mib=1,part_workers=2)
    monkeypatch.setattr(upload,'head',lambda *_:dict(ContentLength=len(s3.object),ETag='"'+s3.etag+'"'))
    payload=b'a'*(1024**2)+b'b'*(1024**2)+b'last'
    state=tmp_path/'parts.json'
    first=upload.MultipartSink(s3,cfg,'archive',state,'source')
    first.write(payload[:1024**2]);first.abandon()
    uploaded=list(s3.calls)
    second=upload.MultipartSink(s3,cfg,'archive',state,'source')
    second.write(payload);receipt=second.finish()
    assert s3.object==payload
    assert s3.calls.count(1)==uploaded.count(1)==1
    assert receipt['bytes']==len(payload) and receipt['sha256']==hashlib.sha256(payload).hexdigest()
    assert receipt['parts']==3


def test_resume_rejects_changed_source(tmp_path):
    s3=FakeS3();cfg=dict(bucket='bucket',part_mib=1,part_workers=1);state=tmp_path/'parts.json'
    first=upload.MultipartSink(s3,cfg,'archive',state,'source');first.write(b'a'*1024**2);first.abandon()
    second=upload.MultipartSink(s3,cfg,'archive',state,'source')
    try:
        with pytest.raises(ValueError,match='disagrees'):second.write(b'b'*1024**2)
    finally:second.abandon()
