#!/usr/bin/env python3
"""Read-only replay of every author-comparison metric; optional new audit receipt.

Intercept the scorer's two output writes, compare the regenerated results, and
verify the original seal as a subset (later reports are separately sealed).
"""
import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from scripts import run_author_pretrained as runner
from scripts.verify_rsi_cycle import require
from tfree_detection.expert_supervision import sha256_file,contained_path,write_new_json

def verify(record=False):
    root=runner.ROOT;out=root/runner.DEFAULT;p=runner.checked(out)
    require(sha256_file(runner.ARCHIVE)==p['archive_sha256'],'Original dense archive changed')
    require(sha256_file(root/'outputs/best_release_20261014_v1/manifest.json')==p['best_release_manifest_sha256'],'Frozen best release changed')
    author=root/'vendor/author_pretrained_20261014_v1/dennler';archive=author/'vgg16.zip'
    meta=runner.read(root/'vendor/author_pretrained_20261014_v1/zenodo-5792917.json')
    published=next(f for f in meta['files'] if f['key']=='vgg16.zip');payload=archive.read_bytes()
    require(len(payload)==published['size'] and 'md5:'+hashlib.md5(payload).hexdigest()==published['checksum'],'Official archive checksum mismatch')
    with ZipFile(archive) as z:
        require(hashlib.sha256(z.read('vgg16/model.h5')).hexdigest()==p['model_sha256'],'Loaded H5 differs from official ZIP')
        for name,r in p['author_split_lists'].items():require(hashlib.sha256(z.read(f'vgg16/{name}.txt')).hexdigest()==r['sha256'],'Author split text differs from ZIP')
    hashes={}
    for t in p['tasks']:
        if t['kind']=='expert':hashes[t['sample']['source_path']]=t['source_sha256']
        elif t['kind']!='dennler':
            for kind in ['image','label']:hashes[t['input'][kind]]=t['input'][kind+'_sha256']
    for name,digest in hashes.items():require(sha256_file(contained_path(root,name))==digest,'Image/label source changed')
    for name,digest in runner.read(out/'seal.json')['files'].items():require(sha256_file(contained_path(out,name))==digest,'Original inference seal mismatch')
    for name,digest in runner.read(out/'comparison_seal.json')['files'].items():require(sha256_file(contained_path(root,name))==digest,'Comparison report seal mismatch')
    writes=[]
    def intercept(path,value):
        if path==out/'results.json':require(value==runner.read(path),'Fresh metric replay differs')
        elif path==out/'seal.json':
            for name,digest in runner.read(path)['files'].items():require(value['files'][name]==digest,'Fresh seal differs')
        else:raise ValueError('Unexpected scoring write')
        writes.append(path.name)
    with patch.object(runner,'write_new_json',intercept),redirect_stdout(StringIO()):runner.score(out)
    require(writes==['results.json','seal.json'],'Score did not regenerate expected outputs')
    result={'all_registered_inferences':len(p['tasks']),'all_metrics_recomputed_identically':True,
        'source_files_rehashed':len(hashes),'official_zip_md5_verified':True,'h5_matches_official_zip':True,
        'author_split_lists_match_official_zip':True,'original_artifacts_modified':False,
        'emseek_pretrained_evaluation_complete':False,'emseek_missing_asset':'author task/general segmentation checkpoint',
        'results_sha256':sha256_file(out/'results.json'),'verifier_sha256':sha256_file(Path(__file__))}
    if record:write_new_json(out/'verification.json',result)
    print(result,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--record',action='store_true');args=a.parse_args();verify(args.record)
