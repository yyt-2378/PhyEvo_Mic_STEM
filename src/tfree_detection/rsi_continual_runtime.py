"""Opt-in execution of a frozen continual-learning research release.

One binding is one fixed seed and outer fold, not an oracle dispatch by material.
Default and any binding failure retain the unmodified native physics result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .expert_runtime import TASK_GEOMETRY
from .expert_student import predict_scores, restore_checkpoint, torch
from .expert_supervision import GEOMETRY_INDEX, normalize_image, sha256_file, write_new_json
from .preprocessing import load_grayscale
from .rsi_continual import retention_gate
from .rsi_feedback import feedback_predictions
from .rsi_runtime import RSIDetection, WARNINGS, inside, pixel_sha256


def read(path):
    return json.loads(path.read_text())


def sealed(root, path):
    value = read(path.with_suffix('.seal.json'))
    if sha256_file(path) != value['sha256']:
        raise ValueError('Research artifact changed')
    for name, digest in value['inputs'].items():
        if sha256_file(inside(root, name)) != digest:
            raise ValueError('Research input changed')
    return read(path)


def create_binding(root, training, run, output):
    root = Path(root).resolve(strict=True)
    training, run, output = [inside(root, p) for p in (training, run, output)]
    protocol = read(training/'protocol.json')
    sealed(root, training/'all_training_sealed.json')
    completion = sealed(root, run/'training_complete.json')
    if protocol.get('format') != 'phyevo-bounded-continual-development-v1':
        raise ValueError('Wrong continual protocol')
    fold = next(f for f in protocol['folds'] if f['fold_id'] == completion['fold_id'])
    if (completion['arm'] not in protocol['arms'] or completion['seed'] not in protocol['seeds']
            or completion['protocol_sha256'] != sha256_file(training/'protocol.json')
            or completion['rounds'] != protocol['rounds']):
        raise ValueError('Invalid completed model family')
    incumbent = {}
    inputs = [training/'protocol.json', training/'all_training_sealed.json', run/'training_complete.json']
    for r in range(1, protocol['rounds']+1):
        path = run/f'round-{r:02d}'/'feedback.json'
        history = sealed(root, path)
        incumbent, decisions = retention_gate(history['control_rows'], incumbent, r, **protocol['gate'])
        if incumbent != history['incumbent'] or decisions != history['decisions'] or history['uses_outer_results']:
            raise ValueError('Controller history does not reproduce')
        inputs.extend([path, path.with_suffix('.seal.json')])
    models = {}
    for geometry, choice in incumbent.items():
        if choice['round'] == 0:
            continue
        checkpoint = run/f'round-{choice["round"]:02d}'/'checkpoint'
        meta = read(checkpoint/'checkpoint.json')
        expected = {k: completion[k] for k in ('arm', 'seed', 'fold_id', 'protocol_sha256')}
        if (any(meta.get(k) != v for k, v in expected.items()) or meta['round'] != choice['round']
                or meta['train_partition'] != fold['partition']['train']
                or fold['partition']['outer_group'] in meta['train_groups']
                or geometry not in meta['geometry_support']
                or sha256_file(checkpoint/'state.npz') != meta['state_sha256']):
            raise ValueError('Selected checkpoint identity mismatch')
        models[geometry] = str(checkpoint.relative_to(root))
        inputs.extend([checkpoint/'checkpoint.json', checkpoint/'state.npz'])
    value = {'format': 'phyevo-continual-research-binding-v1', 'production_approved': False,
             'selection_uses_outer_results': False, 'inference_features': ['geometry'],
             'arm': completion['arm'], 'seed': completion['seed'], 'fold_id': completion['fold_id'],
             'outer_group_excluded': fold['partition']['outer_group'],
             'routes': incumbent, 'models': models,
             'artifacts': {str(p.relative_to(root)): sha256_file(p) for p in inputs},
             'training': str(training.relative_to(root)),
             'evidence_scope': 'same-source-control developmental release, not nested independent inner validation'}
    write_new_json(output, value)
    return value


def load_binding(root, path):
    value = read(inside(root, path))
    if (value.get('format') != 'phyevo-continual-research-binding-v1' or value.get('production_approved') is not False
            or value.get('selection_uses_outer_results') is not False or value.get('inference_features') != ['geometry']):
        raise ValueError('Invalid continual research binding')
    for name, digest in value['artifacts'].items():
        if sha256_file(inside(root, name)) != digest:
            raise ValueError('Changed continual binding artifact')
    protocol_path = inside(root, Path(value['training'])/'protocol.json')
    if str(protocol_path.relative_to(root)) not in value['artifacts']:
        raise ValueError('Unbound training protocol')
    protocol = read(protocol_path)
    for name, digest in protocol['source_hashes'].items():
        if sha256_file(inside(root, name)) != digest:
            raise ValueError('Training producer changed since binding')
    final_paths = [inside(root, p) for p in value['artifacts'] if p.endswith(f'round-{protocol["rounds"]:02d}/feedback.json')]
    if len(final_paths) != 1 or read(final_paths[0])['incumbent'] != value['routes']:
        raise ValueError('Routes differ from frozen controller')
    for geometry, route in value['routes'].items():
        if route['round'] == 0:
            if route['branch'] != 'physics_budgeted':
                raise ValueError('Invalid physics fallback')
            continue
        checkpoint = inside(root, value['models'][geometry])
        for name in ('checkpoint.json', 'state.npz'):
            if str((checkpoint/name).relative_to(root)) not in value['artifacts']:
                raise ValueError('Unbound selected model')
        meta = read(checkpoint/'checkpoint.json')
        if (any(meta[k] != value[k] for k in ('arm', 'seed', 'fold_id')) or meta['round'] != route['round']
                or value['outer_group_excluded'] in meta['train_groups']
                or meta['protocol_sha256'] != sha256_file(protocol_path)):
            raise ValueError('Mixed model family in binding')
    return value


def detect_with_continual(framework, source, *, project_root, binding=None, experimental_research=False,
                         student_device='cpu', **options):
    root = Path(project_root).resolve(strict=True)
    if isinstance(source, (str, Path)):
        source = inside(root, source)
    for key in ('skeleton', 'exemplar', 'sam3_exchange_directory'):
        if isinstance(options.get(key), (str, Path)):
            options[key] = inside(root, options[key])
    physics = framework.detect(source, **options)
    result = RSIDetection(physics, metadata={'format': 'phyevo-continual-research-detection-v1',
        'status': 'physics_default', 'reason': 'not_opted_in', 'production_approved': False,
        'physics_preserved': True, 'coordinate_space': 'source_pixels_xy', 'warnings': WARNINGS.copy()})
    if not experimental_research:
        return result
    try:
        value = load_binding(root, binding)
        geometry = TASK_GEOMETRY.get(physics.task)
        if geometry not in value['routes']:
            raise ValueError('Unsupported continual geometry')
        route = value['routes'][geometry]
        if route['round'] == 0:
            result.metadata.update(status='physics_retained', reason='internal_control_retains_original_physics',
                                   route=route, arm=value['arm'], seed=value['seed'], fold_id=value['fold_id'],
                                   geometry=geometry, binding_sha256=sha256_file(inside(root, binding)))
            return result
        image = load_grayscale(source, frame_index=options.get('frame_index'), projection=options.get('projection'),
                               dataset_path=options.get('dataset_path')) if isinstance(source, Path) else np.asarray(source)
        if image.shape != physics.mask.shape or pixel_sha256(image) != physics.metadata['input_provenance']['pixel_sha256']:
            raise ValueError('Student and physics raw pixels differ')
        values = np.zeros(image.shape, np.float32)
        if route['round'] > 0:
            checkpoint = inside(root, value['models'][geometry])
            model, _, _, _ = restore_checkpoint(checkpoint, device=student_device)
            old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                   torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            try:
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                values = predict_scores(model, normalize_image(image), GEOMETRY_INDEX[geometry], device=student_device,
                                        normalized=True, tile_size=256, overlap=64)
            finally:
                torch.use_deterministic_algorithms(old[0], warn_only=old[1])
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[2:]
            result.student_score = values
        proposed = feedback_predictions(image, physics.mask, [r.to_dict() for r in physics.regions], values, geometry)
        result.candidates = proposed
        result.primary_branch = route['branch']
        result.metadata.update(status='continual_research', reason='frozen_internal_control_route', route=route,
                               arm=value['arm'], seed=value['seed'], fold_id=value['fold_id'], geometry=geometry,
                               binding_sha256=sha256_file(inside(root, binding)), inference_features=['geometry'],
                               checkpoint=value['models'].get(geometry), raw_pixel_sha256=pixel_sha256(image))
    except Exception as exc:
        result.primary_branch = 'physics_native'
        result.candidates, result.student_score = {}, None
        result.metadata.update(status='physics_fallback', reason=str(exc), failure_type=type(exc).__name__)
    return result
