"""Opt-in field storage; frozen studies never import this implementation.

Scientific algorithms, metrics and original worker stay unchanged. Adopting
this runtime in a real study requires an explicit hashed runtime declaration.
"""
import os
from pathlib import Path

import numpy as np

from .contracts import file_hash, inside, write_new
from .field_storage import ArrayPool, exact_child
from .runtime_compat import ConfiguredSandbox
from .stability_state import StabilitySystemExecutor


def storage_binding():
    folder = Path(__file__).parent
    return {name: file_hash(folder / name) for name in
            ('field_storage.py', 'field_storage_worker.py', 'field_storage_runtime.py', 'worker.py')}


class FieldStorageSandbox(ConfiguredSandbox):
    def command(self, program, inputs, output, worker):
        cmd = super().command(program, inputs, output, worker)
        pos = cmd.index(str(worker))
        if cmd[pos - 1:pos + 2] != ['--ro-bind', str(worker), '/worker.py']:
            raise ValueError('Legacy worker binding changed')
        folder = Path(__file__).parent
        cmd[pos] = str(folder / 'field_storage_worker.py')
        settings = ['--ro-bind', str(worker), '/legacy_worker.py',
                    '--ro-bind', str(folder / 'field_storage.py'), '/field_storage.py']
        return cmd[:-4] + settings + cmd[-4:]

    def run(self, program, inputs, output):
        inputs = exact_child(self.workspace, inputs)
        request_path = exact_child(inputs, inputs / 'request.json')
        import json
        request = json.loads(request_path.read_text())
        input_hashes = {'request.json': file_hash(request_path)}
        for item in request.get('samples', []):
            name = item['file']
            if Path(name).name != name:
                raise ValueError('Explicit input basenames required')
            path = exact_child(inputs, inputs / name)
            input_hashes[name] = file_hash(path)
        receipt = super().run(program, inputs, output)
        if any(file_hash(inputs / name) != value for name, value in input_hashes.items()):
            raise ValueError('Staged request or sample manifest changed during execution')
        binding = {'implementation_hashes': storage_binding(), 'predictions_reused': False,
                   'input_request_and_manifest_sha256': input_hashes,
                   'format': 'phyevo-field-references-v1', 'global_pool_visible_to_candidate': False,
                   'scientific_worker_unchanged': True}
        write_new(inside(self.workspace, output) / 'field_storage_runtime.json', binding)
        return receipt | {'field_storage': binding}


class FieldStorageExecutor(StabilitySystemExecutor):
    def __init__(self, root, archive, sandbox, samples, pool, development_transforms,
                 *, max_pool_bytes, minimum_free_bytes):
        if not isinstance(sandbox, FieldStorageSandbox):
            raise ValueError('Field input staging requires the explicitly bound storage runtime')
        super().__init__(root, archive, sandbox, samples, pool, development_transforms)
        self.array_pool = ArrayPool(root, pool, max_pool_bytes=max_pool_bytes,
                                    minimum_free_bytes=minimum_free_bytes)

    def stage(self, folder, cohort, *, mode='predict', steps=0, state=None, branches=None):
        folder = exact_child(self.workspace, folder)
        if mode not in ('predict', 'fit'):
            raise ValueError('Unknown staging mode')
        cohort = tuple(cohort)
        for sample in cohort:
            if any(k in sample.fields for k in ('human', 'known', 'points_xy', 'label')):
                raise ValueError('Labels must not be observable fields')
            if mode == 'fit' and self.protocol.split(sample.group) != 'train':
                raise ValueError('A nontraining source entered fit staging')
        folder.mkdir(parents=True, exist_ok=False)
        records = []
        for index, sample in enumerate(cohort):
            fields = dict(sample.fields)
            if mode == 'fit':
                fields.update(human=sample.human, known=sample.known)
            if branches:
                for name, predictions in branches.items():
                    with np.load(self.checked_ref(predictions[sample.key]), allow_pickle=False) as values:
                        fields['branch_' + name + '_score'] = values['score']
                        fields['branch_' + name + '_support'] = values['support']
            records.append(self.array_pool.stage(folder, f'sample-{index:04d}.fields.json', fields))
        if state:
            source = self.checked_ref(state)
            os.link(source, folder / 'state.npz')
        write_new(folder / 'request.json', {'mode': mode, 'seed': self.protocol.seeds[0],
            'training_steps': steps, 'samples': records, 'state_file': bool(state),
            'device': 'cuda' if self.sandbox.gpu is not None else 'cpu'})
        write_new(folder / 'storage_binding.json', storage_binding())
