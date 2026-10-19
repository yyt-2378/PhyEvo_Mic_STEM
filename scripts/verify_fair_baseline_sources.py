#!/usr/bin/env python3
"""Verify downloaded upstream blobs against pinned Git object IDs, read-only."""
import hashlib
import json
from pathlib import Path
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json

ROOT=Path(__file__).resolve().parents[1]
BASE=contained_path(ROOT,'vendor/fair_baselines_20261014')

def main():
    result={}
    for name in ('emseek','dennler'):
        tree=json.loads((BASE/name/'upstream_tree.json').read_text());rows=[]
        for item in tree['tree']:
            relative=item['path']
            if item['type']!='blob' or not (relative.endswith('.py') or relative in ('LICENSE','README.md','temdefect_env.yml')):continue
            path=contained_path(ROOT,BASE/name/relative);data=path.read_bytes()
            git_hash=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if git_hash!=item['sha']:raise ValueError(f'Pinned source mismatch: {relative}')
            rows.append({'path':str(path.relative_to(ROOT)),'git_blob_sha1':git_hash,'sha256':sha256_file(path)})
        result[name]={'commit':tree['sha'],'verified_blobs':rows}
    model=BASE/'dennler/vgg16.zip'
    result['released_weight_download']={'path':str(model.relative_to(ROOT)),'bytes':model.stat().st_size,
        'expected_bytes':134380387,'md5':hashlib.md5(model.read_bytes()).hexdigest(),
        'expected_md5':'c130bb1ffecba7903213bb7d246065f5','used_for_training':False,
        'status':'truncated_download_retained_not_extracted_or_used'}
    write_new_json(BASE/'verified_source_provenance.json',result)
    print({k:len(v['verified_blobs']) for k,v in result.items() if 'verified_blobs' in v})

if __name__=='__main__':main()
