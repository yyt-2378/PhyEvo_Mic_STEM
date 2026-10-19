#!/usr/bin/env python3
"""Mark the exact prohibited-source experiment ineligible, without deleting it."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    for name in ('fair_baselines_dense_20261014_v1','fair_baselines_dense_20261014_v2'):
        directory=contained_path(ROOT,ROOT/'outputs'/name)
        protocol=json.loads((directory/'protocol.json').read_text())
        training=[r['id'] for r in protocol['rows'] if r['split']=='train']
        if len(training)!=37:raise ValueError('Unexpected quarantine target')
        checkpoints={str(p.relative_to(directory)):sha256_file(p) for p in directory.glob('runs/*/*/state.pt')}
        with ZipFile(directory/'disqualified_producer_source.zip','x',compression=ZIP_DEFLATED) as archive:
            archive.write(ROOT/'scripts/run_fair_dense.py','scripts/run_fair_dense.py')
        write_new_json(directory/'DISQUALIFIED.json',{
            'status':'DISQUALIFIED_NOT_EVIDENCE_DO_NOT_LOAD_OR_PROMOTE',
            'reason':'The registered Dennler archive is locked_external_test and explicitly forbidden for training. This attempt incorrectly trained on 37 images.',
            'source_policy':'datasets/manifest.yaml: dataset dennler_2021_hrstem_defects labels.note',
            'policy_file_sha256':sha256_file(ROOT/'datasets/manifest.yaml'),
            'protocol_sha256':sha256_file(directory/'protocol.json'),'prohibited_training_image_ids':training,
            'checkpoints_quarantined_by_status':checkpoints,
            'excluded_from_all_result_tables_and_manuscript_claims':True,
            'files_deleted_or_modified':[],'original_dataset_or_historical_results_changed':False,
            'guarded_at_training_and_evaluation_entry':True})
        print(name,'disqualified checkpoints',len(checkpoints))

if __name__=='__main__':main()
