"""Storage-only input decoder around the unmodified scientific worker."""
import importlib.util
import runpy

import numpy as np

spec = importlib.util.spec_from_file_location('field_storage_codec', '/field_storage.py')
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
original_load = np.load
np.load = codec.input_loader(original_load, '/input')
try:
    runpy.run_path('/legacy_worker.py', run_name='__main__')
finally:
    np.load = original_load
