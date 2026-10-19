#!/usr/bin/env python3
"""Download pinned public baseline assets; never execute or overwrite them."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'vendor/fair_baselines_20261014'
REPOS = {'emseek': ('PEESEgroup/EMSeek', 'e46afc5f3199f87a385fd25bd7930e4bf2ccb8ca'),
         'dennler': ('nkdnnlr/TEMDefectClassification', '72d37c3ca5a84cfa39337c7243a80e3ffcc0f648')}

def checked(path):
    path = path.resolve()
    path.relative_to(ROOT)
    return path

def download(url, path, md5=None):
    path = checked(path)
    checked(path.parent).mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    request = urllib.request.Request(url, headers={'User-Agent': 'PhyEvo-Mic-reproducibility-audit'})
    with urllib.request.urlopen(request, timeout=120) as source, path.open('xb') as target:
        while chunk := source.read(1024*1024):
            target.write(chunk)
    payload = path.read_bytes()
    if md5 and hashlib.md5(payload).hexdigest() != md5:
        raise ValueError(f'Checksum mismatch, retained for audit: {path}')
    print(f'{path.relative_to(ROOT)} {len(payload)} bytes', flush=True)
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(), 'md5': hashlib.md5(payload).hexdigest()}

def main():
    checked(DEST.parent)
    if DEST.exists():
        raise FileExistsError('Use a new asset version; existing downloads remain protected')
    records = []
    for label, (repo, revision) in REPOS.items():
        record = download(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1', DEST / label / 'upstream_tree.json')
        records.append(record)
        tree = json.loads((ROOT / record['path']).read_text())
        for item in tree['tree']:
            name = item['path']
            if item['type'] == 'blob' and (name.endswith('.py') or name in ('LICENSE', 'README.md', 'temdefect_env.yml')):
                records.append(download(f'https://raw.githubusercontent.com/{repo}/{revision}/{name}', DEST / label / name))
    records.append(download('https://zenodo.org/api/records/5792917/files/vgg16.zip/content',
                            DEST / 'dennler/vgg16.zip', 'c130bb1ffecba7903213bb7d246065f5'))
    with zipfile.ZipFile(DEST / 'dennler/vgg16.zip') as archive:
        inventory = [{'name': z.filename, 'bytes': z.file_size} for z in archive.infolist()]
        for item in archive.infolist():
            if item.is_dir():
                continue
            if item.filename.startswith('/') or '..' in Path(item.filename).parts:
                raise ValueError('Unsafe archive entry')
            path = checked(DEST / 'dennler/released_model' / item.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(archive.read(item))
    with (DEST / 'provenance.json').open('x') as stream:
        json.dump({'repositories': REPOS, 'downloads': records, 'dennler_model_inventory': inventory,
                   'download_only_no_upstream_execution': True}, stream, indent=2)

if __name__ == '__main__':
    main()
