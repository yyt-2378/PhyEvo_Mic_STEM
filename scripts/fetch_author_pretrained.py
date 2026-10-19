#!/usr/bin/env python3
"""Fetch public author artifacts with checked byte ranges and immutable receipts.

No upstream code executes. Existing inputs and partial previous downloads remain
untouched. EMSeek's large data archive is inventoried by its central directory;
it is not called a fully checksum-verified download.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'vendor/author_pretrained_20261014_v1'

def checked(path):
    p=Path(path).resolve();p.relative_to(ROOT)
    if p==ROOT:raise ValueError('Project root is not an output')
    return p

def put(path,payload):
    path=checked(path);checked(path.parent).mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError(f'Existing artifact differs: {path}')
    else:
        with path.open('xb') as f:f.write(payload)

def js(path,value):put(path,(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode())

def request(url,headers=None,limit=16*1024*1024):
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'PhyEvo-Mic-author-weight-audit',**(headers or {})})
            with urllib.request.urlopen(req,timeout=45) as r:
                payload=r.read(limit+1)
                if len(payload)>limit:raise ValueError('Unexpected oversized response')
                return payload,r.status,dict(r.headers)
        except Exception:
            if attempt==3:raise
            time.sleep(2*(attempt+1))

def metadata(record):
    cached=DEST/f'zenodo-{record}.json'
    if cached.exists():return json.loads(cached.read_text())
    url=f'https://zenodo.org/api/records/{record}'
    data,_,_=request(url);value=json.loads(data)
    js(DEST/f'zenodo-{record}.json',value)
    return value

def byte_range(url,start,end,total):
    for attempt in range(6):
        try:
            data,status,headers=request(url+f'?download=1&audit_range={start}-{end}&attempt={attempt}',
                {'Range':f'bytes={start}-{end}'},end-start+1)
            h={k.lower():v for k,v in headers.items()}
            if status!=206 or h.get('content-range')!=f'bytes {start}-{end}/{total}' or len(data)!=end-start+1:
                raise ValueError(f'HTTP range mismatch: status={status}, range={h.get("content-range")}, bytes={len(data)}')
            return data
        except Exception as e:
            print('range retry',start,end,attempt,str(e),flush=True)
            if attempt==5:raise
            time.sleep(attempt+1)

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.position=0;self.receipts=[]
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset+(0 if whence==0 else self.position if whence==1 else self.size)
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def read(self,size=-1):
        size=min(self.size-self.position,size if size>=0 else self.size-self.position)
        if size<=0:return b''
        if size>16*1024*1024:raise ValueError('Remote inventory read unexpectedly large')
        start=self.position;data=byte_range(self.url,start,start+size-1,self.size);self.position+=size
        self.receipts.append({'start':start,'size':size,'sha256':hashlib.sha256(data).hexdigest()})
        return data

def download(file,label):
    size=file['size'];url=file['links']['self'];chunk=4*1024*1024
    base=DEST/label;checked(base).mkdir(parents=True,exist_ok=True)
    def part(start):
        end=min(size,start+chunk)-1;path=base/'parts'/f'{start:012d}-{end:012d}.part';receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists():
            data=path.read_bytes();r=json.loads(receipt.read_text())
            if len(data)!=end-start+1 or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Part changed')
        else:
            data=byte_range(url,start,end,size);put(path,data)
            js(receipt,{'url':url,'start':start,'end':end,'total':size,'sha256':hashlib.sha256(data).hexdigest()})
        print(label,'part',start,end,flush=True);return path
    with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(part,range(0,size,chunk)))
    archive=base/file['key']
    if not archive.exists():
        with checked(archive).open('xb') as f:
            for p in parts:f.write(p.read_bytes())
    payload=archive.read_bytes();md5=hashlib.md5(payload).hexdigest()
    if len(payload)!=size or file['checksum']!='md5:'+md5:raise ValueError('Whole-archive checksum mismatch')
    js(base/'verified_download.json',{'url':url,'size':size,'md5':md5,'sha256':hashlib.sha256(payload).hexdigest()})
    return archive

def inventory(archive):
    return [{'name':z.filename,'bytes':z.file_size,'compressed_bytes':z.compress_size,'CRC':z.CRC} for z in archive.infolist()]

def dennler():
    meta=metadata('5792917');file=next(f for f in meta['files'] if f['key']=='vgg16.zip')
    path=download(file,'dennler')
    with ZipFile(path) as archive:
        js(path.parent/'inventory.json',inventory(archive))
        for item in archive.infolist():
            if item.is_dir():continue
            if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts:raise ValueError('Unsafe zip member')
            put(path.parent/'released'/item.filename,archive.read(item))
    print('DENNLER VERIFIED',path,flush=True)

def emseek():
    meta=metadata('18340509')
    code=next(f for f in meta['files'] if f['key']=='EMSeek.zip');path=download(code,'emseek')
    with ZipFile(path) as archive:
        js(path.parent/'code_inventory.json',inventory(archive))
        for item in archive.infolist():
            if item.is_dir() or not item.filename.lower().endswith(('.md','.txt','.py','.yaml','.yml','.json')):continue
            if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts:raise ValueError('Unsafe zip member')
            put(path.parent/'published_code'/item.filename,archive.read(item))
    data=next(f for f in meta['files'] if f['key']=='EM_data.zip')
    remote=RemoteZip(data['links']['self'],data['size'])
    with ZipFile(remote) as archive:items=inventory(archive)
    weight_names=[r for r in items if r['name'].lower().endswith(('.pth','.pt','.h5','.hdf5','.safetensors','.ckpt','.onnx','.pkl','.bin'))]
    js(path.parent/'data_archive_inventory.json',{'url':data['links']['self'],'published_size':data['size'],
        'published_checksum':data['checksum'],'whole_archive_downloaded':False,'range_receipts':remote.receipts,
        'members':items,'weight_like_members':weight_names})
    print('EMSEEK INVENTORY',len(items),'weight-like files',weight_names,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('target',choices=['dennler','emseek']);a=p.parse_args()
    (dennler if a.target=='dennler' else emseek)()
