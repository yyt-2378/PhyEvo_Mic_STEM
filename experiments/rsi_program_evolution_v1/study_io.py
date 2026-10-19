"""Load prepared RSI inputs without importing a historical study driver.

The manifest, array and engine checks are the original real_study contract.
No input preparation, fitting, output creation or historical paths live here.
"""
import json
from pathlib import Path

import numpy as np

from .contracts import Protocol, engine_fingerprint, file_hash, inside
from .executor import Sample, sample_fingerprint


ROOT = Path(__file__).resolve().parents[2]


def load_study(directory, *, root=None):
    root = ROOT if root is None else Path(root).resolve()
    directory = inside(root, directory)
    protocol_values = json.loads(inside(root, directory / "protocol.json").read_text())
    protocol = Protocol(**{k: tuple(v) if k.endswith("_groups") or k == "seeds" else v
                           for k, v in protocol_values.items()})
    if protocol.engine_version != engine_fingerprint():
        raise ValueError("Study engine changed: do not resume under changed execution/evaluation code")
    dataset = json.loads(inside(root, directory / "dataset.json").read_text())
    samples = []
    for entry in dataset["samples"]:
        path = inside(root, directory / entry["file"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Source checksum mismatch: " + str(path.relative_to(root)))
        with np.load(path, allow_pickle=False) as arrays:
            values = {key: arrays[key] for key in arrays.files}
        supervision = {k: values.pop(k) for k in ("human", "known", "points_xy")}
        samples.append(Sample(entry["key"], entry["group"], entry["geometry"], values, **supervision))
    if sample_fingerprint(samples) != protocol.data_version:
        raise ValueError("Prepared input data changed")
    return directory, protocol, samples
