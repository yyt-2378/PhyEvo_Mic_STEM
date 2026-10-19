#!/usr/bin/env python3
"""Recover publicly shared author weights from a pinned historical README.

No credentials, Drive API keys, upstream code execution, or overwritten files.
The current README lost this link; keep its historical Git blob as provenance.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'vendor/author_pretrained_emseek_20261014_v2'
COMMIT = '87369f029dfacf814d6acaa4e125a3ef290dad4a'
README_BLOB = '72ab01e0dc361ffb03295357f882880c537c30ee'
README_URL = f'https://raw.githubusercontent.com/PEESEgroup/EMSeek/{COMMIT}/README.md'
ROOT_FOLDER = '1ltlPT8bclLc9QXfSOEWyKm64ZdtSPxQM'
PRETRAINED_FOLDER = '1QH9GrUsU8yzPq3a_EFac9-v48XkICLyA'
SEGMENTOR_FOLDER = '1Z6IrYENc5i9HM_AXZTs06I8eCj630PkT'
EXPECTED = {
    'task1.pth': '1FEpYeb6yxil9y7l_itxdlT7GdmoJN_UP',
    'task2.pth': '1UR2rfNX2agTPVExahcOMJOL7f5cMyIJa',
    'task3.pth': '1Uukxjq9YR8sYQ8LU8z2e1ag8H_Vv1RNv',
    'task4.pth': '187GJGILPNKbJsO86fLrh4sD9wqHeckJG',
    'task5.pth': '1poHPgTSaCYVgrTbId8SGnz98aSPVBsyG',
    'unet_ref.pth': '1Ng_1tTxQopyM1AK1FVuvz0sn6Yts-3ll',
}


def put(path, payload):
    path = Path(path).resolve()
    path.relative_to(DEST.resolve())
    path.parent.resolve().relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def put_json(path, value):
    put(path, (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode())


def get(url, limit):
    req = urllib.request.Request(url, headers={'User-Agent': 'PhyEvo-Mic-author-weight-audit'})
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError('Response exceeded expected size')
        return payload, response.headers.get('Content-Type', '')


def parse_folder(html):
    match = re.search(r"window\['_DRIVE_ivd'\] = '([^']+)';", html)
    if match is None:
        raise ValueError('No public Drive inventory; do not authenticate or bypass access')
    # Decode JS string escapes, then parse JSON, never evaluate JavaScript.
    escaped = match.group(1).replace('\\/', '/')
    data = json.loads(escaped.encode().decode('unicode_escape'))
    return [{'id': row[0], 'name': row[2], 'mime': row[3], 'size': row[13]}
            for row in (data[0] or [])]


def folder(identifier):
    url = f'https://drive.google.com/drive/folders/{identifier}?hl=en'
    payload, _ = get(url, 4 * 1024 * 1024)
    return {'url': url, 'entries': parse_folder(payload.decode()),
            'response_sha256': hashlib.sha256(payload).hexdigest()}


def download(entry):
    name, identifier, size = entry['name'], entry['id'], int(entry['size'])
    if EXPECTED.get(name) != identifier or size != 110942962:
        raise ValueError('Published file identity/size differs from discovered inventory')
    path = DEST / 'SegMentor' / name
    receipt_path = path.with_suffix('.receipt.json')
    url = f'https://drive.usercontent.google.com/download?id={identifier}&export=download&confirm=t'
    if path.exists():
        if not receipt_path.exists():
            raise ValueError('Existing weight lacks provenance; preserved, not overwritten')
        payload = path.read_bytes()
        receipt = json.loads(receipt_path.read_text())
        if receipt['sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError('Existing author weight changed')
    else:
        payload, content_type = get(url, size)
        if 'html' in content_type or not payload.startswith(b'PK\x03\x04'):
            raise ValueError('Download is not a PyTorch archive')
    if len(payload) != size:
        raise ValueError('Truncated author weight download')
    with ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError('Author weight archive CRC failed')
    receipt = {'name': name, 'drive_id': identifier, 'url': url, 'size': size,
               'sha256': hashlib.sha256(payload).hexdigest(), 'zip_crc_verified': True,
               'author_published_checksum_available': False, 'readme_commit': COMMIT}
    put(path, payload)
    put_json(receipt_path, receipt)
    print('VERIFIED AUTHOR WEIGHT', name, size, receipt['sha256'], flush=True)
    return receipt


def main():
    DEST.resolve().relative_to(ROOT)
    readme, _ = get(README_URL, 1024 * 1024)
    blob = hashlib.sha1(f'blob {len(readme)}\0'.encode() + readme).hexdigest()
    if blob != README_BLOB or ROOT_FOLDER.encode() not in readme:
        raise ValueError('Historical official README identity mismatch')
    put(DEST / 'historical_README.md', readme)
    inventories = [folder(item) for item in [ROOT_FOLDER, PRETRAINED_FOLDER, SEGMENTOR_FOLDER]]
    if not any(e['id'] == PRETRAINED_FOLDER and e['name'] == 'pretrained' for e in inventories[0]['entries']):
        raise ValueError('Published pretrained folder link changed')
    if not any(e['id'] == SEGMENTOR_FOLDER and e['name'] == 'SegMentor' for e in inventories[1]['entries']):
        raise ValueError('Published segmentation folder link changed')
    entries = inventories[2]['entries']
    if {e['name']: e['id'] for e in entries} != EXPECTED:
        raise ValueError('Unexpected public weight inventory')
    # HTML contains anonymous session metadata: retain only parsed public asset inventory.
    receipt = {'official_readme_url': README_URL, 'readme_git_blob': blob,
               'readme_sha256': hashlib.sha256(readme).hexdigest(),
               'folders': [{k: v for k, v in item.items() if k != 'response_sha256'} for item in inventories]}
    put_json(DEST / 'discovery.json', receipt)
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(download, entries))
    put_json(DEST / 'weights.json', {'readme_commit': COMMIT, 'weights': receipts})


if __name__ == '__main__':
    main()
