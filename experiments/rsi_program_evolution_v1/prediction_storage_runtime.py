"""Opt-in field inputs plus newly computed, content-shared prediction outputs."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .contracts import file_hash, write_new
from .field_storage import ArrayBundle, array_identity, exact_child
from .field_storage_runtime import FieldStorageSandbox, storage_binding
from .prediction_storage import PredictionPool


def prediction_binding():
    folder = Path(__file__).parent
    return storage_binding() | {name: file_hash(folder / name) for name in
        ('prediction_storage.py', 'prediction_storage_worker.py', 'prediction_storage_runtime.py')}


@dataclass(frozen=True)
class SharedPredictionSandbox(FieldStorageSandbox):
    prediction_pool: PredictionPool | None = None
    max_written_prediction_bytes: int = 256 * 1024**2

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.prediction_pool, PredictionPool) or self.max_written_prediction_bytes <= 0:
            raise ValueError('Explicit prediction pool and fresh-output budget required')
        exact_child(self.workspace, self.prediction_pool.root)

    def command(self, program, inputs, output, worker):
        cmd = super().command(program, inputs, output, worker)
        folder = Path(__file__).parent
        original = folder / 'field_storage_worker.py'
        pos = cmd.index(str(original))
        cmd[pos] = str(folder / 'prediction_storage_worker.py')
        settings = ['--ro-bind', str(original), '/field_worker.py',
                    '--ro-bind', str(folder / 'prediction_storage.py'), '/prediction_storage.py',
                    '--ro-bind', str(output.with_name(output.name + '.storage') / 'config.json'),
                    '/prediction_storage_config.json']
        return cmd[:-4] + settings + cmd[-4:]

    def run(self, program, inputs, output):
        inputs = exact_child(self.workspace, inputs)
        output = exact_child(self.workspace, output)
        if output.exists():
            raise FileExistsError('Prediction output must be a new directory')
        request = json.loads((inputs / 'request.json').read_text())
        sample_keys = []
        if request['mode'] == 'predict':
            for item in request['samples']:
                if Path(item['file']).name != item['file']:
                    raise ValueError('Explicit sample basename required')
                path = exact_child(inputs, inputs / item['file'])
                loader = ArrayBundle(path, np.load) if path.suffix == '.json' else np.load(path, allow_pickle=False)
                with loader as arrays:
                    sample_keys.append(array_identity(arrays['image'])[2])
        known = self.prediction_pool.snapshot(sample_keys)
        folder = exact_child(self.workspace, output.with_name(output.name + '.storage'))
        folder.mkdir(parents=True, exist_ok=False)
        config = {'enabled': request['mode'] == 'predict', 'known_hashes': known,
                  'max_prediction_bytes': 256 * 1024**2,
                  'max_written_prediction_bytes': self.max_written_prediction_bytes}
        write_new(folder / 'config.json', config)
        config_hash = file_hash(folder / 'config.json')
        execution = super().run(program, inputs, output)
        if file_hash(folder / 'config.json') != config_hash:
            raise ValueError('Prediction storage snapshot changed during execution')
        sharing = None
        if execution['status'] == 'ok' and request['mode'] == 'predict':
            sharing = self.prediction_pool.finalize(output, sample_keys, known)
        receipt = {'implementations': prediction_binding(), 'snapshot_sha256': config_hash,
                   'current_input_image_scope_only': True, 'candidate_sees_only_hashes_not_global_arrays': True,
                   'actual_inference_required': True, 'sharing': sharing}
        write_new(output / 'prediction_storage_runtime.json', receipt)
        return execution | {'prediction_storage': receipt}
