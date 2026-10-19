"""Read existing prediction NPZ members without repacking their array payloads.

Only explicitly selected, same-case score/support pairs enter each job. Existing
files are never replaced. New input paths are read-only sandbox hardlinks.
This file also serves as the field decoder inside the isolated worker.
"""
import importlib.util
import json
import os
from pathlib import Path
import re

import numpy as np

if __package__:
    from . import field_storage as base
else:
    spec = importlib.util.spec_from_file_location('base_field_codec', '/base_field_storage.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)


class MemberBundle(base.ArrayBundle):
    def __getitem__(self, key):
        ref = next((r for r in self.fields if r['name'] == key), None)
        if ref is None or 'member' not in ref:
            return super().__getitem__(key)
        if (ref['member'] not in ('score', 'support')
                or not re.fullmatch(r'[a-f0-9]{64}', ref['sha256'])
                or ref['file'] != ref['sha256'] + '.npz'):
            raise ValueError('Invalid prediction member reference')
        path = base.exact_child(self.path.parent, self.path.parent / 'arrays' / ref['file'])
        if base.sha256(path) != ref['sha256']:
            raise ValueError('Borrowed prediction bytes changed')
        with self.loader(path, allow_pickle=False) as values:
            if set(values.files) != {'score', 'support'}:
                raise ValueError('Only score/support may enter borrowed input files')
            value, metadata, identity = base.array_identity(values[ref['member']])
        if metadata != ref['array'] or identity != ref['identity']:
            raise ValueError('Borrowed prediction array changed')
        return value


def input_loader(legacy_load, input_root):
    input_root = Path(input_root)
    def load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and str(file).endswith('.fields.json'):
            path = base.exact_child(input_root, Path(file))
            if (path.parent != input_root or args or kwargs != {'allow_pickle': False}
                    or not re.fullmatch(r'sample-\d{4}\.fields\.json', path.name)):
                raise ValueError('Only the trusted explicit input manifest is supported')
            return MemberBundle(path, legacy_load)
        return legacy_load(file, *args, **kwargs)
    return load


def stage_members(pool, folder, name, fields, branch_sources):
    """Validate entire prediction pairs before exposing any of their members."""
    folder = base.exact_child(pool.workspace, folder)
    if not re.fullmatch(r'sample-\d{4}\.fields\.json', name):
        raise ValueError('Explicit input manifest basename required')
    destination = base.exact_child(folder, folder / name)
    if destination.exists():
        raise FileExistsError('Input manifest already exists')
    prepared = []
    for branch, source in branch_sources.items():
        if not re.fullmatch(r'b[0-9]+', branch):
            raise ValueError('Anonymous branch name required')
        source = base.exact_child(pool.workspace, source)
        if source.stat().st_dev != folder.stat().st_dev:
            raise ValueError('Borrowed inputs require the same filesystem; no silent copy')
        digest = base.sha256(source)
        with np.load(source, allow_pickle=False) as values:
            if set(values.files) != {'score', 'support'}:
                raise ValueError('Only score/support may enter borrowed input files')
            score, support = values['score'], values['support']
            if (score.shape != fields['image'].shape or support.shape != score.shape
                    or score.dtype != np.float32 or support.dtype != np.bool_):
                raise ValueError('Aligned canonical prediction pair required')
            for member in ('score', 'support'):
                _, metadata, identity = base.array_identity(values[member])
                prepared.append((source, {'name': f'branch_{branch}_{member}', 'member': member,
                    'file': digest + '.npz', 'sha256': digest, 'identity': identity, 'array': metadata}))
    entries = []
    arrays = base.exact_child(folder, folder / 'arrays'); arrays.mkdir(exist_ok=True)
    for key, value in fields.items():
        if any(ref['name'] == key for _, ref in prepared):
            raise ValueError('Ambiguous branch field')
        ref = pool.put(value)
        source = pool.root / ref['file']; target = base.exact_child(arrays, arrays / ref['file'])
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError('Unexpected staged array')
        else:
            os.link(source, target)
        entries.append({'name': key, **ref})
    for source, ref in prepared:
        target = base.exact_child(arrays, arrays / ref['file'])
        if target.exists():
            if base.sha256(target) != ref['sha256']:
                raise ValueError('Unexpected borrowed prediction')
        else:
            os.link(source, target)
        entries.append(ref)
    base.new_json(destination, {'format': base.FORMAT, 'fields': entries,
                               'required_decoder': 'prediction-member-v1'})
    return {'file': name}


if __package__:
    from .contracts import file_hash, write_new
    from .field_storage_runtime import FieldStorageExecutor
    from .prediction_storage import PredictionPool
    from .prediction_storage_runtime import SharedPredictionSandbox

    class MemberSandbox(SharedPredictionSandbox):
        def command(self, program, inputs, output, worker):
            cmd = super().command(program, inputs, output, worker)
            original = Path(__file__).with_name('field_storage.py')
            pos = cmd.index(str(original))
            assert cmd[pos - 1:pos + 2] == ['--ro-bind', str(original), '/field_storage.py']
            cmd[pos] = str(Path(__file__))
            return cmd[:-4] + ['--ro-bind', str(original), '/base_field_storage.py'] + cmd[-4:]

        def run(self, program, inputs, output):
            receipt = super().run(program, inputs, output)
            binding = {'extension_sha256': file_hash(Path(__file__)), 'prediction_members_read_only': True,
                       'old_files_replaced': 0, 'science_worker_unchanged': True}
            write_new(output / 'member_storage_runtime.json', binding)
            return receipt | {'member_storage': binding}

    class MemberExecutor(FieldStorageExecutor):
        def stage(self, folder, cohort, *, mode='predict', steps=0, state=None, branches=None):
            folder = base.exact_child(self.workspace, folder)
            if mode not in ('fit', 'predict'):
                raise ValueError('Unknown staging mode')
            cohort = tuple(cohort)
            for sample in cohort:
                if any(k in sample.fields for k in ('human', 'known', 'points_xy', 'label')):
                    raise ValueError('Labels must not be observable fields')
                if mode == 'fit' and self.protocol.split(sample.group) != 'train':
                    raise ValueError('Nontraining source entered fit staging')
            folder.mkdir(parents=True, exist_ok=False)
            records = []
            for index, sample in enumerate(cohort):
                fields = dict(sample.fields)
                if mode == 'fit':
                    fields.update(human=sample.human, known=sample.known)
                sources = {name: self.checked_ref(predictions[sample.key])
                           for name, predictions in (branches or {}).items()}
                records.append(stage_members(self.array_pool, folder, f'sample-{index:04d}.fields.json', fields, sources))
            if state:
                os.link(self.checked_ref(state), folder / 'state.npz')
            write_new(folder / 'request.json', {'mode': mode, 'seed': self.protocol.seeds[0],
                'training_steps': steps, 'samples': records, 'state_file': bool(state),
                'device': 'cuda' if self.sandbox.gpu is not None else 'cpu'})
            write_new(folder / 'member_storage_binding.json', {'extension_sha256': file_hash(Path(__file__))})

    def adapt_runner(old, directory, policy):
        pool = PredictionPool(old.workspace, directory / 'predictions',
            max_pool_bytes=policy['max_prediction_pool_bytes'], minimum_free_bytes=policy['minimum_free_bytes'])
        sandbox = MemberSandbox(old.workspace, old.sandbox.runtime, timeout=old.sandbox.timeout,
            gpu=old.sandbox.gpu, prediction_pool=pool,
            max_written_prediction_bytes=policy['max_written_prediction_bytes'])
        return MemberExecutor(old.workspace, old.archive, sandbox, old.samples, directory / 'fields',
            old.development_transforms, max_pool_bytes=policy['max_input_pool_bytes'],
            minimum_free_bytes=policy['minimum_free_bytes'])
