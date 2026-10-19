"""Export the accepted active portfolio without resurrecting discarded research.

Source archives remain read-only. This is a new executable packaging, not a new
scientific gain. Numerical equivalence must be demonstrated before using it.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

from .contracts import canonical, digest, file_hash, write_new
from .field_storage import exact_child

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZED_SOURCE = Path('/data2/yaotian/phyevo_mic_rsi_runs/rsi_stable_expansion_20261017_v4/full-s41')
ACCEPTED_HASH = 'be8e7ecb17f84f15e0f544ebb2ca97807c0364862e1c6cdf82c7345771b93b06'

WRAPPER = '''"""Trusted lazy execution of an immutable, explicitly listed portfolio."""
from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path
import numpy as np

CONFIG=json.loads(Path('/program/portfolio.json').read_text())
MODULES={}
WEIGHTS={}
COUNTER=0

def module(name, filename):
    if name not in MODULES:
        spec=importlib.util.spec_from_file_location(name, '/program/'+filename)
        obj=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        MODULES[name]=obj
    return MODULES[name]

def weights(name, filename):
    if name not in WEIGHTS:
        if filename is None:
            WEIGHTS[name]={}
        else:
            with np.load('/program/'+filename, allow_pickle=False) as z:
                WEIGHTS[name]={k:z[k] for k in z.files}
    return WEIGHTS[name]

class Branches(Mapping):
    def __init__(self, sample, context):
        self.sample=sample
        self.context=context
        self.cache={}
    def __iter__(self):
        return iter(CONFIG['active_branches'])
    def __len__(self):
        return len(CONFIG['active_branches'])
    def __getitem__(self, name):
        if name not in CONFIG['branches']:
            raise KeyError(name)
        if name not in self.cache:
            b=CONFIG['branches'][name]
            result=module(name,b['program']).predict(self.sample,weights(name,b['checkpoint']),self.context)
            score=np.asarray(result['score'])
            support=np.asarray(result['support'])
            if score.shape!=self.sample['image'].shape or support.shape!=score.shape:
                raise ValueError('Branch output shape changed')
            if not np.isfinite(score).all() or support.dtype!=np.bool_:
                raise ValueError('Invalid branch output')
            # Match the original isolated worker's serialization boundary.
            self.cache[name]={'score':score.astype(np.float32),'support':support}
        return self.cache[name]

def predict(sample,state,context):
    global COUNTER
    branches=Branches(sample,context)
    choice=module('router','router.py').route(sample,branches,weights('router',CONFIG['router_checkpoint']),context)
    if choice.get('branch') not in CONFIG['active_branches'] or not isinstance(choice.get('reason'),str):
        raise ValueError('Invalid retained router choice')
    result=branches[choice['branch']]
    with open('/work/route-%04d.json'%COUNTER,'x') as f:
        json.dump(choice,f)
    with open('/work/computed-%04d.json'%COUNTER,'x') as f:
        json.dump({'computed':list(branches.cache)},f)
    COUNTER+=1
    return result
'''


def export(workspace, episode, result_relative, expected_result_hash, destination,
           *, maximum_copy_bytes=16 * 1024**2):
    workspace = Path(workspace).resolve(strict=True)
    episode = Path(episode).absolute()
    if episode.resolve(strict=True) != episode or not (
            episode.is_relative_to(workspace) or episode == AUTHORIZED_SOURCE):
        raise ValueError('Explicit workspace or authorized read-only source required')
    destination = exact_child(workspace, destination)
    if destination.exists():
        raise FileExistsError('Use an exclusive new portfolio destination')
    if not isinstance(maximum_copy_bytes, int) or maximum_copy_bytes <= 0:
        raise ValueError('Positive copy ceiling required')
    source_hashes = {}

    def checked(relative, expected=None):
        p = exact_child(episode, episode / relative)
        h = file_hash(p)
        if expected is not None and h != expected:
            raise ValueError('Source hash mismatch: ' + relative)
        source_hashes[relative] = h
        return p

    result_path = checked(result_relative, expected_result_hash)
    result = json.loads(result_path.read_text())
    if (result.get('status') != 'accepted' or result.get('decision', {}).get('accepted') is not True
            or result.get('actual_execution') is not True):
        raise ValueError('Actually executed accepted system required')
    candidate = result['candidate']
    if not re.fullmatch(r'[a-f0-9]{64}', candidate):
        raise ValueError('Invalid accepted identity')
    prefix = f'archive/programs/{candidate}/'
    state = json.loads(checked(prefix + 'system_state.json').read_text())
    spec = json.loads(checked(prefix + 'system_spec.json', state['specification_sha256']).read_text())
    if state['candidate'] != candidate:
        raise ValueError('Frozen state belongs to a different candidate')
    router = checked(prefix + 'router.py')
    if digest(router.read_text()) != spec['router_sha256']:
        raise ValueError('Router binding changed')
    active = spec['active_branches']
    if (not isinstance(active, list) or not active or len(active) != len(set(active))
            or any(not re.fullmatch(r'b[0-9]+', n) for n in active)):
        raise ValueError('Explicit unique active branch names required')
    copies = [('router.py', router)]
    branches = {}
    for name in active:
        binding = spec['library'][name]
        cap = json.loads(checked(binding['path'], binding['sha256']).read_text())
        program = checked(cap['program']['path'], cap['program']['sha256'])
        copies.append((name + '.py', program))
        checkpoint = None
        if cap['checkpoint']:
            checkpoint = name + '.npz'
            ref = cap['checkpoint']
            copies.append((checkpoint, checked(ref['path'], ref['sha256'])))
        branches[name] = {'program': name + '.py', 'checkpoint': checkpoint,
                          'source_binding': binding, 'weight_training_groups': cap['weight_training_groups']}
    router_checkpoint = None
    if state['router_checkpoint']:
        router_checkpoint = 'router_state.npz'
        ref = state['router_checkpoint']
        copies.append((router_checkpoint, checked(ref['path'], ref['sha256'])))
    total = sum(p.stat().st_size for _, p in copies) + len(WRAPPER.encode())
    if total > maximum_copy_bytes:
        raise ValueError('Portfolio copy exceeds declared storage ceiling')
    destination.mkdir(parents=True, exist_ok=False)
    program = destination / 'program'; program.mkdir()
    exported_hashes = {}
    for name, source in copies:
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != source_hashes[str(source.relative_to(episode))]:
            raise ValueError('Source changed during export')
        target = program / name
        with target.open('xb') as f:
            f.write(payload)
        exported_hashes[name] = file_hash(target)
    with (program / 'pipeline.py').open('x') as f:
        f.write(WRAPPER)
    configuration = {'format': 'accepted-active-portfolio-v1', 'source_candidate': candidate,
                     'active_branches': active, 'branches': branches,
                     'router_checkpoint': router_checkpoint}
    write_new(program / 'portfolio.json', configuration)
    for name in ('pipeline.py', 'portfolio.json'):
        exported_hashes[name] = file_hash(program / name)
    if any(file_hash(episode / p) != h for p, h in source_hashes.items()):
        raise ValueError('Source changed during export')
    manifest = {'status': 'exported_pending_numerical_replay', 'new_scientific_gain': False,
                'source_episode': str(episode), 'source_candidate': candidate,
                'source_accepted_result': result_relative, 'source_hashes': source_hashes,
                'exported_hashes': exported_hashes, 'active_branches': active,
                'historical_library_count': len(spec['library']),
                'inactive_branch_records_not_resurrected': sorted(set(spec['library']) - set(active)),
                'copied_program_and_state_bytes': total, 'maximum_copy_bytes': maximum_copy_bytes,
                'source_unchanged': True, 'array_datasets_copied': False,
                'wrapper_semantics': 'Unchanged router, on-demand active branch execution, original float32 boundary',
                'equivalence_not_assumed': True, 'production_promoted': False}
    write_new(destination / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = export(ROOT, AUTHORIZED_SOURCE, 'requests/r06/result.json', ACCEPTED_HASH,
                    args.destination)
    print(canonical({k: result[k] for k in ('status', 'source_candidate', 'active_branches',
                                         'copied_program_and_state_bytes', 'source_unchanged')}), flush=True)


if __name__ == '__main__':
    main()
