"""Optional prediction serializer; keeps the old scientific worker unchanged."""
import importlib.util
import json
from pathlib import Path
import runpy

spec = importlib.util.spec_from_file_location('prediction_codec', '/prediction_storage.py')
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
config = json.loads(Path('/prediction_storage_config.json').read_text())
original_open = Path.open
writer = codec.PredictionWriter(config, '/work', original_open)
Path.open = lambda self, mode='r', *args, **kwargs: writer.open(self, mode, *args, **kwargs)
try:
    runpy.run_path('/field_worker.py', run_name='__main__')
    writer.finish()
finally:
    Path.open = original_open
