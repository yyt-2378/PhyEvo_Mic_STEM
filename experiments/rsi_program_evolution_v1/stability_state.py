"""Additive pre-admission stability execution for a future declared study.

The sealed v3 study does not import this module. Development transforms are
explicitly visible research evidence, never called independent acquisitions.
Fitting still sees only the original train cohort. Programs and weights are
unchanged during a probe, and no function here selects a system for deployment.
"""
from collections import Counter
from dataclasses import replace
import fcntl
import os

import numpy as np

from .contracts import file_hash, inside, write_new
from .evaluation import evaluate, summarize
from .expansion_state import CompactSystemExecutor, array_digest
from .stability_diagnostics import TRANSFORMS, transform_array
from .system_state import read


class StabilitySystemExecutor(CompactSystemExecutor):
    def __init__(self, root, archive, sandbox, samples, pool, development_transforms,
                 readonly_input_pools=()):
        if (not development_transforms or len(development_transforms) != len(set(development_transforms))
                or not set(development_transforms) <= set(TRANSFORMS)):
            raise ValueError('Distinct declared development transforms required')
        self.development_transforms = tuple(development_transforms)
        super().__init__(root, archive, sandbox, samples, pool)
        self.readonly_input_pools = tuple(inside(self.workspace, p) for p in readonly_input_pools)
        if any(not p.is_dir() or p == self.pool for p in self.readonly_input_pools):
            raise ValueError('Existing distinct read-only input pools required')

    def stage(self, folder, cohort, *, mode='predict', steps=0, state=None, branches=None):
        """Reuse identical immutable inputs, not predictions or measured gains.

        Only new pool entries and hard links are created. Existing pool bytes,
        markers and directories are never written. The actual program execution
        still happens in the same read-only-input sandbox as before.
        """
        folder = inside(self.workspace, folder)
        if folder.exists():
            raise FileExistsError('Input staging requires a new directory')
        cohort = tuple(cohort)
        for sample in cohort if self.readonly_input_pools else ():
            fields = dict(sample.fields)
            if any(k in fields for k in ('human', 'known', 'points_xy', 'label')):
                raise ValueError('Labels must not be observable fields')
            if mode == 'fit':
                if self.protocol.split(sample.group) != 'train':
                    raise ValueError('A nontraining source entered fit staging')
                fields.update(human=sample.human, known=sample.known)
            elif mode != 'predict':
                raise ValueError('Unknown staging mode')
            if branches:
                for name, predictions in branches.items():
                    with np.load(self.checked_ref(predictions[sample.key]), allow_pickle=False) as values:
                        fields['branch_' + name + '_score'] = values['score']
                        fields['branch_' + name + '_support'] = values['support']
            key = array_digest(fields)
            target = self.pool / (key + '.npz'); marker = target.with_suffix('.json')
            with target.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if marker.exists():
                    continue  # Superclass verifies the content-bound entry.
                if target.exists():
                    raise FileExistsError('Incomplete pool entry preserved; not overwritten')
                for source_pool in self.readonly_input_pools:
                    source = inside(self.workspace, source_pool / target.name)
                    source_marker = source.with_suffix('.json')
                    if not source_marker.is_file():
                        continue
                    expected = {'arrays': key, 'sha256': file_hash(source)}
                    if read(source_marker) != expected:
                        raise ValueError('Immutable source input pool entry changed')
                    os.link(source, target)
                    write_new(marker, expected)
                    write_new(target.with_suffix('.reuse.json'), {
                        'source': str(source.relative_to(self.workspace)), 'sha256': expected['sha256'],
                        'source_marker_sha256': file_hash(source_marker), 'predictions_reused': False})
                    break
        return super().stage(folder, cohort, mode=mode, steps=steps, state=state, branches=branches)

    def cohorts(self, split):
        if not split.startswith('stability-'):
            return super().cohorts(split)
        transform = split.removeprefix('stability-')
        if transform not in self.development_transforms:
            raise ValueError('Transform was not declared for development')
        result = []
        for sample in super().cohorts('development'):
            shape = sample.fields['image'].shape
            if sample.geometry != 'interface' or any(v.shape != shape for v in sample.fields.values()):
                raise ValueError('Stability probes require aligned scalar interface fields')
            move = lambda a: np.ascontiguousarray(transform_array(a, transform))
            result.append(replace(sample, key=f'{transform}-' + sample.key,
                fields={k: move(v) for k, v in sample.fields.items()}, human=move(sample.human),
                known=move(sample.known), points_xy=np.empty((0, 2))))
        return result

    def probe_system(self, identity, transform):
        """Actual frozen-state execution, with separately bound smoke receipts.

        Never overload ordinary validation trials: existing `latest` callers
        must keep returning the original development evaluation. A cached probe
        must match its own recorded event and current executable-state binding.
        """
        split = 'stability-' + transform
        cohort = self.cohorts(split)
        folder, spec = self.system(identity)
        state = self.frozen_state(identity)['router_checkpoint']
        run = self.root / 'system_jobs' / (identity + '-' + split)
        result_path = run / 'result.json'
        if result_path.exists():
            events = [e['body'] for e in self.archive.events('development_stability_measured')
                      if e['body']['candidate'] == identity and e['body']['transform'] == transform]
            if len(events) != 1 or events[0]['receipt']['sha256'] != file_hash(result_path):
                raise ValueError('Stability receipt is unbound or changed')
            receipt = read(result_path)
            if (receipt.get('router_checkpoint') != state
                    or receipt['system_specification_sha256'] != file_hash(folder / 'system_spec.json')):
                raise ValueError('Stability receipt belongs to a different executable state')
            for reference in receipt.get('predictions', {}).values():
                self.checked_ref(reference)
            return events[0]['trial'], receipt
        run.mkdir(exist_ok=False)
        try:
            branches = {name: self.predict_capability(spec['library'][name], split)['predictions']
                        for name in spec['active_branches']}
            self.stage(run / 'input', cohort, state=state, branches=branches)
            execution = self.sandbox.run(folder, run / 'input', run / 'output')
            if execution['status'] != 'ok':
                raise RuntimeError((run / 'output/stderr.log').read_text()[-4000:])
            rows, predictions, choices = [], {}, Counter()
            for index, sample in enumerate(cohort):
                path = run / 'output' / f'prediction-{index:04d}.npz'
                route = read(run / 'output' / f'route-{index:04d}.json')
                if route['branch'] not in branches:
                    raise ValueError('Unknown selected branch')
                with np.load(path, allow_pickle=False) as actual:
                    with np.load(self.checked_ref(branches[route['branch']][sample.key]), allow_pickle=False) as expected:
                        if any(not np.array_equal(actual[k], expected[k]) for k in ('score', 'support')):
                            raise ValueError('Stability router modified a capability output')
                    measured = evaluate(sample.validation_target(), dict(actual), self.protocol)
                choices[route['branch']] += 1
                rows.append({'case': sample.key, 'group': sample.group, 'seed': self.protocol.seeds[0],
                             'role': self.protocol.split(sample.group), 'branch': route['branch'], **measured})
                predictions[sample.key] = self.ref(path)
            receipt = {'rows': rows, 'summary': summarize(rows), 'predictions': predictions,
                       'choices': dict(choices), 'execution': execution, 'actual_execution': True}
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            receipt = {'error': str(error), 'actual_execution': True}
        receipt.update(transform=transform, router_checkpoint=state, router_training=None,
            system_specification_sha256=file_hash(folder / 'system_spec.json'),
            cohort_semantics='declared development stability probe; not a sealed audit or new data')
        write_new(result_path, receipt)
        trial = self.archive.record_trial(identity, 'smoke', 'ok' if 'rows' in receipt else 'failed', receipt)
        self.archive.event('development_stability_measured', {'candidate': identity, 'transform': transform,
            'trial': trial, 'receipt': self.ref(result_path)})
        return trial, receipt


def stability_admission(quality_decision, original_rows, probes, required_transforms,
                        maximum_regression, retained_best):
    """Conjoin (never replace) the existing gain gate with whole-system retention.

    A stable but non-improving candidate still fails the gain gate. Retention
    compares each source against both its own original score and the best
    previously accepted score on the *same* declared transform. Neither missing
    executions nor mismatched budgets can count as a successful check.
    """
    if (not required_transforms or len(required_transforms) != len(set(required_transforms))
            or not set(required_transforms) <= set(TRANSFORMS)
            or not np.isfinite(maximum_regression) or maximum_regression < 0):
        raise ValueError('Declared transforms and finite nonnegative tolerance required')
    if set(probes) != set(required_transforms) or set(retained_best) != set(required_transforms):
        raise ValueError('Every declared transform needs measured evidence and a retention reference')
    original = {(r['group'], r['case'], r['seed']): r for r in original_rows}
    if not original or len(original) != len(original_rows):
        raise ValueError('Nonempty unique original cohort required')
    original_summary = summarize(original_rows)
    regressions, failures = [], []
    for transform in required_transforms:
        receipt = probes[transform]
        if 'rows' not in receipt or not receipt.get('actual_execution'):
            failures.append(transform)
            continue
        if (receipt.get('transform') != transform or receipt.get('router_training') is not None
                or not isinstance(receipt.get('execution'), dict)
                or receipt['execution'].get('status') != 'ok'):
            raise ValueError('Expected successful frozen-state transform execution without refitting')
        transformed = {(r['group'], r['case'], r['seed']): r for r in receipt['rows']}
        expected = {(g, transform + '-' + c, seed) for g, c, seed in original}
        if len(transformed) != len(receipt['rows']) or set(transformed) != expected:
            raise ValueError('Transform cohort does not match original cases and sources')
        for (group, case, seed), row in original.items():
            other = transformed[group, transform + '-' + case, seed]
            for record in (row, other):
                if (any(record[k] != row[k] for k in ('budget', 'denominator', 'metric'))
                        or not 0 <= record['coverage'] <= 1 or record['selected'] > record['budget']):
                    raise ValueError('Invalid or changed metric, denominator or budget')
        measured = summarize(receipt['rows'])
        if set(retained_best[transform]) != set(original_summary):
            raise ValueError('Every source needs its prior retained transform score')
        for group, value in measured.items():
            prior = retained_best[transform][group]
            if not np.isfinite(prior) or not 0 <= prior <= 1:
                raise ValueError('Invalid retained transform score')
            own = original_summary[group]['mean']
            if value['mean'] < max(own, prior) - maximum_regression:
                regressions.append({'transform': transform, 'group': group,
                    'transformed': value['mean'], 'own_original': own, 'prior_retained_best': prior})
    stable = not failures and not regressions
    return {'accepted': bool(quality_decision['accepted'] and stable),
            'quality_decision': quality_decision, 'stability_passed': stable,
            'failed_executions': failures, 'stability_regressions': regressions,
            'reason': 'gain_and_stability_retained' if quality_decision['accepted'] and stable
                      else 'gain_or_stability_not_satisfied'}
