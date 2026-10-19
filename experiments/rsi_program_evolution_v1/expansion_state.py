"""Additive capability-expansion evidence and compact immutable input staging.

The old execution/evaluation engine remains unchanged. Shared input bundles are
hard-linked only into read-only sandbox mounts; inference never receives labels.
The per-case library ceiling is an offline diagnostic, never an inference rule.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import fcntl
import hashlib
import os
from pathlib import Path

import numpy as np

from .contracts import digest, file_hash, inside, write_new
from .evaluation import summarize
from .system_state import SystemExecutor, joint_admission, read


def array_digest(arrays):
    records = {}
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError('Finite numeric arrays required')
        records[name] = {'shape': value.shape, 'dtype': str(value.dtype),
                         'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    return digest(records)


class CompactSystemExecutor(SystemExecutor):
    """Deduplicate immutable bundles without deleting or changing old artifacts."""
    def __init__(self, root, archive, sandbox, samples, pool):
        super().__init__(root, archive, sandbox, samples)
        self.pool = inside(self.workspace, pool)
        self.pool.mkdir(parents=True, exist_ok=True)

    def stage(self, folder, cohort, *, mode='predict', steps=0, state=None, branches=None):
        folder = inside(self.workspace, folder)
        folder.mkdir(parents=True, exist_ok=False)
        records = []
        for index, sample in enumerate(cohort):
            fields = dict(sample.fields)
            if any(k in fields for k in ('human', 'known', 'points_xy', 'label')):
                raise ValueError('Labels must not be part of observable fields')
            if mode == 'fit':
                if self.protocol.split(sample.group) != 'train':
                    raise ValueError('A nontraining source entered fit staging')
                fields.update(human=sample.human, known=sample.known)
            elif mode != 'predict':
                raise ValueError('Unknown staging mode')
            if branches:
                for name, predictions in branches.items():
                    with np.load(self.checked_ref(predictions[sample.key]), allow_pickle=False) as z:
                        fields['branch_' + name + '_score'] = z['score']
                        fields['branch_' + name + '_support'] = z['support']
            key = array_digest(fields)
            bundle = inside(self.workspace, self.pool / (key + '.npz'))
            marker = bundle.with_suffix('.json')
            with bundle.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if marker.exists():
                    if read(marker) != {'arrays': key, 'sha256': file_hash(bundle)}:
                        raise ValueError('Shared input bundle changed')
                else:
                    # An interrupted partial bundle is preserved, never overwritten.
                    with bundle.open('xb') as stream:
                        np.savez_compressed(stream, **fields)
                    write_new(marker, {'arrays': key, 'sha256': file_hash(bundle)})
                name = f'sample-{index:04d}.npz'
                destination = inside(self.workspace, folder / name)
                os.link(bundle, destination)
            records.append({'file': name})
        if state:
            source = self.checked_ref(state)
            with (folder / 'state.npz').open('xb') as stream:
                stream.write(source.read_bytes())
        write_new(folder / 'request.json', {'mode': mode, 'seed': self.protocol.seeds[0],
            'training_steps': steps, 'samples': records, 'state_file': bool(state),
            'device': 'cuda' if self.sandbox.gpu is not None else 'cpu'})

    def import_capability(self, source_runner, binding):
        """Copy small definitions/states, preserving their exact frozen bytes."""
        source = source_runner.checked_ref(binding)
        original = read(source)
        origin = {'path': str(source.relative_to(self.workspace)), 'sha256': file_hash(source)}
        identity = digest({'imported_frozen_capability': origin})
        target = self.root / 'capabilities' / identity
        target.mkdir(exist_ok=False)
        (target / 'program').mkdir()
        program = source_runner.checked_ref(original['program'])
        with (target / 'program/pipeline.py').open('xb') as stream:
            stream.write(program.read_bytes())
        checkpoint = None
        if original['checkpoint']:
            source_state = source_runner.checked_ref(original['checkpoint'])
            with (target / 'state.npz').open('xb') as stream:
                stream.write(source_state.read_bytes())
            checkpoint = self.ref(target / 'state.npz')
        write_new(target / 'definition.json', {'imported_from': origin,
            'program_sha256': file_hash(program), 'weights_refitted': False,
            'original_definition_sha256': original['definition_sha256']})
        write_new(target / 'capability.json', {'identity': identity,
            'definition_sha256': file_hash(target / 'definition.json'),
            'program': self.ref(target / 'program/pipeline.py'), 'checkpoint': checkpoint,
            'training': original['training'], 'weight_training_groups': original['weight_training_groups'],
            'data_version': self.protocol.data_version, 'imported_from': origin})
        return self.ref(target / 'capability.json')

    def import_system(self, source_runner, identity, extras=()):
        original_folder, _ = source_runner.system(identity)
        state = source_runner.frozen_state(identity)
        library = {name: self.import_capability(source_runner, binding)
                   for name, binding in state['library'].items()}
        for extra_runner, binding in extras:
            name = 'b' + str(max(int(n[1:]) for n in library) + 1)
            library[name] = self.import_capability(extra_runner, binding)
        router_state = None
        if state['router_checkpoint']:
            path = self.root / 'imported_router_state.npz'
            with path.open('xb') as stream:
                stream.write(source_runner.checked_ref(state['router_checkpoint']).read_bytes())
            router_state = self.ref(path)
        spec = {'library': library, 'active_branches': state['active_branches'],
            'parent_system': None, 'router_parent_state': router_state, 'router_training_steps': 0,
            'operation': 'exact_frozen_best_complete_system_import',
            'origin_system': identity,
            'origin_state_sha256': file_hash(original_folder / 'system_state.json')}
        return self._register((original_folder / 'router.py').read_text(), spec, None)


def library_gaps(current_rows, branch_rows):
    """Separate selection regret from a library's residual detection deficit."""
    current = {r['case']: r for r in current_rows}
    if len(current) != len(current_rows) or not branch_rows:
        raise ValueError('Unique cases and nonempty capability library required')
    indexed = {name: {r['case']: r for r in rows} for name, rows in branch_rows.items()}
    for name, rows in branch_rows.items():
        if len(indexed[name]) != len(rows) or indexed[name].keys() != current.keys():
            raise ValueError('Unmatched capability cohorts')
    output = []
    for key, row in current.items():
        values = {}
        for name, rows in indexed.items():
            candidate = rows[key]
            if any(candidate[k] != row[k] for k in ('group', 'metric', 'budget', 'denominator')):
                raise ValueError('Unmatched metric, source, denominator or candidate budget')
            values[name] = candidate['coverage']
        best = max(values.values())
        budget_ceiling = min(1., row['budget'] / max(1, row['denominator']))
        if any(not 0. <= v <= budget_ceiling + 1e-12 for v in values.values()):
            raise ValueError('Coverage exceeds the matched candidate-budget upper bound')
        output.append({'case': key, 'group': row['group'], 'current_coverage': row['coverage'],
            'best_existing_branch_coverage': best, 'branch_coverage': values,
            'routing_gap': max(0., best - row['coverage']),
            'candidate_budget_coverage_upper_bound': budget_ceiling,
            'remaining_library_deficit': max(0., budget_ceiling - best),
            'unavoidable_budget_limited_miss_fraction': 1. - budget_ceiling,
            'oracle_is_diagnostic_only': True})
    return output


def selected_novel_gain(rows, frozen_branch_rows, frozen_names):
    """New branches must beat the old per-case selection ceiling to add capacity."""
    gaps = library_gaps(rows, frozen_branch_rows)
    indexed = {row['case']: row for row in rows}
    groups = defaultdict(list)
    novel_calls = defaultdict(int)
    for gap in gaps:
        row = indexed[gap['case']]
        is_new = row['branch'] not in frozen_names
        groups[row['group']].append(row['coverage'] - gap['best_existing_branch_coverage'] if is_new else 0.)
        novel_calls[row['group']] += int(is_new)
    return {'mean_selected_new_gain_over_old_casewise_ceiling':
                {g: float(np.mean(v)) for g, v in groups.items()},
            'new_branch_calls': dict(novel_calls),
            'ceiling_is_not_a_deployable_baseline': True}


def expansion_admission(new, old, base, protocol, retained_best):
    decision = joint_admission(new, old, base, protocol, retained_best)
    new_summary, old_summary = summarize(new), summarize(old)
    validation = list(protocol.validation_groups)
    if not validation or not set(validation) <= new_summary.keys():
        raise ValueError('Source-held-out development evaluation is required')
    gain = float(np.mean([new_summary[g]['mean'] - old_summary[g]['mean'] for g in validation]))
    decision['weight_fit_excluded_development_gain'] = gain
    decision['accepted'] = bool(decision['accepted'] and gain >= protocol.minimum_gain)
    decision['reason'] = 'whole_system_and_nontraining_source_gain' if decision['accepted'] else 'insufficient_gain_or_retention'
    return decision


def transformed_audit(samples):
    """Held-back equivariance checks, explicitly NOT independent acquisitions."""
    result = []
    for sample in samples:
        transform = lambda a: np.ascontiguousarray(np.rot90(a, 1))
        if sample.geometry != 'interface' or any(v.shape != sample.fields['image'].shape for v in sample.fields.values()):
            raise ValueError('Stress transform currently requires aligned scalar interface fields')
        result.append(replace(sample, key='stress-' + sample.key, group='stress-' + sample.group,
            fields={k: transform(v) for k, v in sample.fields.items()}, human=transform(sample.human),
            known=transform(sample.known), points_xy=np.empty((0, 2))))
    return result
