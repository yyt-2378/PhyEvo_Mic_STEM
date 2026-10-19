"""Opt-in library configuration inside an unchanged OS sandbox.

The sealed active-direction study keeps the original Sandbox. New callers can
use this adapter; it changes no mount, credential, network or device access.
"""
from pathlib import Path

from .contracts import file_hash, inside, write_new
from .configured_sandbox import ConfiguredSandbox


def replay_failed_import(study, trajectory, round_number, output):
    """Post-seal infrastructure check, never a replacement research round."""
    from dataclasses import replace
    import json
    from .active_study import EPISODES, load
    from .archive import Archive
    from .executor import Executor
    from .real_study import ROOT, read
    study=inside(ROOT,study); output=inside(ROOT,output)
    if not (study/'results.json').is_file():
        raise ValueError('Complete and seal all original searches before compatibility replay')
    if output.exists():raise FileExistsError('Replay requires a new evidence destination')
    _,_,episode,original,old_executor=load(study,trajectory)
    try:
        result=read(episode/'requests'/f'r{round_number:02d}'/'result.json')
        if result['status']!='execution_failed' or 'Could not determine home directory' not in result.get('error',''):
            raise ValueError('Only the identified import/configuration failure can use this replay')
        candidate=original.candidate(result['candidate'])['proposal']
        if candidate.training_steps:raise ValueError('This infrastructure retry must not retrain')
        source_tail=original.verify()
        expected=next(t['archive_tail'] for t in read(study/'results.json')['trajectories'] if t['trajectory']==trajectory)
        if source_tail!=expected:raise ValueError('Original sealed archive changed')
        protocol=original.protocol; samples=old_executor.samples
    finally:original.close()
    output.mkdir(parents=True,exist_ok=False)
    provenance={'scope':'post-seal infrastructure-only replay; excluded from search, admission and RSI gain counts',
                'study_results_sha256':file_hash(study/'results.json'),'source_archive_tail':source_tail,
                'source_candidate':candidate.identity,'pipeline_is_unchanged':True,
                'adapter_sha256':file_hash(Path(__file__)),'trajectory':trajectory,'round':round_number}
    write_new(output/'provenance.json',provenance)
    archive=Archive(output/'archive',protocol,create=True)
    executor=Executor(archive,ConfiguredSandbox(ROOT,ROOT/'.venvs/sam2',timeout=protocol.timeout_seconds,gpu=EPISODES[trajectory]),samples)
    try:
        replay=replace(candidate,action='generate',parents=(),
                       observation='Post-seal unchanged-program replay in sandbox-local library configuration')
        assert replay.pipeline_code==candidate.pipeline_code
        identity=archive.add(replay); trial,receipt=executor.execute(identity)
        write_new(output/'result.json',{'candidate':identity,'trial':trial,'summary':receipt.get('summary'),
            'error':receipt.get('error'),'status':'ok' if 'rows' in receipt else 'failed',
            'source_pipeline_sha256':file_hash(episode/'archive/programs'/candidate.identity/'pipeline.py'),
            'replayed_pipeline_sha256':file_hash(output/'archive/programs'/identity/'pipeline.py'),
            'selected_for_production':False,'included_in_original_rsi_results':False,'archive_tail':archive.verify()})
        print(json.dumps(read(output/'result.json')),flush=True)
    finally:archive.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Post-seal unchanged-program configuration replay')
    parser.add_argument('--study',required=True)
    parser.add_argument('--trajectory',required=True,choices=('t01','t02','t03'))
    parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    replay_failed_import(args.study,args.trajectory,args.round,args.output)
