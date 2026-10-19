"""Post-seal diagnostics from saved executions; never rerun or reselect a model.

Only completed episode pairs may be read. Named source IDs and pixel arrays are
kept out of the research feedback. The packet is historical development evidence,
not a new held-out evaluation and not evidence of a completed repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np

from .contracts import digest, file_hash, inside, write_new
from .stability_diagnostics import field_information, ordinal_tie_probe, prediction_stability


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def bound(episode, reference):
    path = inside(episode, reference['path'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Bound stability evidence changed')
    return path


def compare_saved_receipts(episode, original, stress, fraction, aliases):
    originals = {r['case']: r for r in original['rows']}
    moved = {r['case']: r for r in stress['rows']}
    if (len(originals) != len(original['rows']) or len(moved) != len(stress['rows'])
            or set(moved) != {'stress-' + key for key in originals}
            or set(original['predictions']) != set(originals)
            or set(stress['predictions']) != set(moved)):
        raise ValueError('Unmatched original and transformed execution cohorts')
    records = []
    for index, (key, row) in enumerate(originals.items()):
        changed = moved['stress-' + key]
        if changed['group'] != 'stress-' + row['group']:
            raise ValueError('Transformed source does not match original source')
        a_path = bound(episode, original['predictions'][key])
        b_path = bound(episode, stress['predictions']['stress-' + key])
        with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
            diagnostic = prediction_stability(a['score'], a['support'], b['score'], b['support'],
                'rot90', fraction, original_branch=row['branch'], transformed_branch=changed['branch'])
        records.append({'sample': f'observation-{index:03d}', 'source': aliases[row['group']],
            'original_branch': row['branch'], 'transformed_branch': changed['branch'],
            'existing_label_side_coverage': {'original': row['coverage'], 'transformed': changed['coverage'],
                'change': changed['coverage'] - row['coverage']},
            'observable_stability': diagnostic,
            'original_prediction_sha256': original['predictions'][key]['sha256'],
            'transformed_prediction_sha256': stress['predictions']['stress-' + key]['sha256']})
    groups = {}
    for alias in aliases.values():
        members = [r for r in records if r['source'] == alias]
        if not members:
            continue
        st = [r['observable_stability'] for r in members]
        groups[alias] = {'observations': len(members),
            'changed_branch_count': sum(not r['same_selected_branch'] for r in st),
            'mean_coverage_change': float(np.mean([r['existing_label_side_coverage']['change'] for r in members])),
            'mean_selection_changed_fraction_of_image': float(np.mean([r['selection_changed_fraction_of_image'] for r in st])),
            'original_cutoff_split_tie_count': sum(r['original_cutoff']['cutoff_splits_equal_values'] for r in st)}
    return {'groups': groups, 'observations': records,
            'original_receipt_sha256': digest(original), 'stress_receipt_sha256': digest(stress)}


def field_packet(episode, prepared, aliases):
    """Use the shared frozen base's prediction inputs, never the label store."""
    cap_path = bound(episode, prepared['baseline_library']['b0'])
    receipt_path = cap_path.parent / 'development/predictions.json'
    inputs = cap_path.parent / 'development/input'
    request_path = inputs / 'request.json'
    request, receipt = read(request_path), read(receipt_path)
    if request['mode'] != 'predict' or len(request['samples']) != len(receipt['rows']):
        raise ValueError('Expected aligned label-free prediction inputs')
    observations = []
    for index, (sample, row) in enumerate(zip(request['samples'], receipt['rows'])):
        path = inside(ROOT, inputs / sample['file'])
        with np.load(path, allow_pickle=False) as values:
            if {'human', 'known'} & set(values.files):
                raise ValueError('Labels are not allowed in observable-field diagnostics')
            valid = values['valid']
            fields = {}
            for name in ('base', 'physics', 'student', 'variation'):
                value = values[name]
                fields[name + '_within_valid'] = {'information': field_information(value, valid),
                    'legacy_ordinal_probe': ordinal_tie_probe(value, valid)}
                if name == 'physics':
                    physical = valid & values['physics_valid']
                    fields['physics_within_physical_valid'] = {
                        'information': field_information(value, physical),
                        'legacy_ordinal_probe': ordinal_tie_probe(value, physical)}
        observations.append({'sample': f'observation-{index:03d}', 'source': aliases[row['group']],
            'input_sha256': file_hash(path), 'fields': fields})
    return {'observations': observations, 'request_sha256': file_hash(request_path),
            'receipt_sha256': file_hash(receipt_path), 'raw_labels_read': False}


def episode_report(study, episode_name, config):
    if episode_name not in {f'{arm}-s{seed}' for arm in ('route', 'full') for seed in (23, 41, 67)}:
        raise ValueError('Explicit known episode required')
    episode = inside(study, episode_name)
    prepared_path = episode / 'prepared.json'; prepared = read(prepared_path)
    connection = sqlite3.connect((episode / 'archive/archive.sqlite3').as_uri() + '?mode=ro', uri=True)
    try:
        seals = [json.loads(r[0]) for r in connection.execute("select body from events where kind='search_sealed'")]
        if len(seals) != 1:
            raise ValueError('Episode search must already be sealed')
        selected = seals[0]['system']['executable_system']
        trials = [(r[0], r[1], r[2], json.loads(r[3])) for r in connection.execute(
            "select id,candidate,split,receipt from trials where split in ('validation','audit')")]
    finally:
        connection.close()
    sources = config['evaluation']['train_groups'] + config['evaluation']['validation_groups']
    aliases = {name: f'D{i}' for i, name in enumerate(sources)}
    systems = {}
    for role, candidate in {'frozen': prepared['baseline'], 'selected': selected}.items():
        pair = {}
        for split in ('validation', 'audit'):
            matching = [r for r in trials if r[1] == candidate and r[2] == split]
            if len(matching) != 1 or 'rows' not in matching[0][3]:
                raise ValueError(f'Completed unique {role} {split} execution required')
            pair[split] = matching[0]
        systems[role] = dict(compare_saved_receipts(episode, pair['validation'][3], pair['audit'][3],
                config['evaluation']['area_fraction'], aliases),
            candidate=candidate, development_trial=pair['validation'][0], stress_trial=pair['audit'][0])
    fields = field_packet(episode, prepared, aliases)
    expected = [(r['sample'], r['source']) for r in fields['observations']]
    if any([(r['sample'], r['source']) for r in value['observations']] != expected for value in systems.values()):
        raise ValueError('Field and prediction observations differ in order')
    return {'episode': episode_name, 'search_checkpoint': seals[0], 'systems': systems,
            'field_evidence': fields, 'prepared_sha256': file_hash(prepared_path)}


def report(study, output, episodes):
    study, output = inside(ROOT, study), inside(ROOT, output)
    if output.exists():
        raise FileExistsError('New diagnostic output required; prior evidence is immutable')
    config_path = study / 'protocol.json'; config = read(config_path)
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError('Distinct explicit episodes required')
    results = [episode_report(study, episode, config) for episode in episodes]
    packet = {'scope': 'post-seal historical stability diagnosis; not repair or new held-out evaluation',
        'completed_episodes_examined': episodes, 'all_six_episodes_examined': len(episodes) == 6,
        'study_protocol_sha256': file_hash(config_path),
        'diagnostic_source_sha256': file_hash(Path(__file__).with_name('stability_diagnostics.py')),
        'report_source_sha256': file_hash(__file__),
        'programs_modified': False, 'production_replaced': False, 'admission_changed': False,
        'raw_images_or_annotations_exported': False, 'selection_metric_unchanged': True,
        'metric': 'known-positive coverage at fixed 10% entire-image budget, not accuracy or IoU',
        'whole_program_causal_ablation_completed': False, 'episodes': results}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'diagnostics.json', packet)
    print(json.dumps({'output': str(output / 'diagnostics.json'),
        'episodes': {e['episode']: {r: v['groups'] for r, v in e['systems'].items()} for e in results}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study'); parser.add_argument('output')
    parser.add_argument('--episode', action='append', required=True)
    args = parser.parse_args()
    report(args.study, args.output, args.episode)
