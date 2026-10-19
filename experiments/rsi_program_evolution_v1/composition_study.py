"""Prospective bounded whole-program RSI from an explicitly configured system.

Only new files in the declared scope; no candidate imports in the controller.
The scientific proposal comes from an independent model, not this harness.
"""
import argparse
import ast
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np

from .contracts import Protocol, digest, engine_fingerprint, file_hash, verify_snapshot, write_new
from .evaluation import evaluate, summarize
from .expansion_state import expansion_admission
from .field_storage import ArrayPool, exact_child
from .prediction_storage import PredictionPool
from .prediction_storage_runtime import SharedPredictionSandbox
from .study_io import load_study
from .stability_diagnostics import transform_array
from .stability_state import stability_admission

ROOT = Path(__file__).resolve().parents[2]
SCOPE = None
SOURCE = None
ALIASES = {}
RUN_CONFIGURATION = None
ENTRY = "import sys\nsys.path.insert(0, '/program')\nfrom capabilities import predict, fit\n"
FIELDS = ('image', 'base', 'support', 'valid', 'physics', 'student', 'physics_valid',
          'variation', 'domain_labels')


def configure_run(path):
    """Resolve locations only; never create a directory or load experiment assets."""
    global SCOPE, SOURCE, ALIASES, RUN_CONFIGURATION
    path = exact_child(ROOT, Path(path) if Path(path).is_absolute() else ROOT / path)
    value = json.loads(path.read_text())
    required = {'scope', 'source', 'source_diagnostics', 'physical_source', 'runtime', 'source_aliases'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Run configuration requires exactly: ' + ', '.join(sorted(required)))
    resolved = {}
    for key in required - {'source_aliases'}:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError('Explicit run path required: ' + key)
        raw = Path(value[key])
        resolved[key] = exact_child(ROOT, raw if raw.is_absolute() else ROOT / raw)
    scope = resolved['scope']
    for key, input_path in resolved.items():
        if key != 'scope' and (scope == input_path or scope.is_relative_to(input_path)
                               or input_path.is_relative_to(scope)):
            raise ValueError('Run scope must be separate from input and runtime paths')
    aliases = value['source_aliases']
    if (not isinstance(aliases, dict) or len(aliases) != 3
            or any(not isinstance(k, str) or not k for k in aliases)
            or set(aliases.values()) != {'D0', 'D1', 'D2'}):
        raise ValueError('Declare three unique source aliases D0, D1, D2')
    # Assign only after every field passed, so a failed configuration is atomic.
    RUN_CONFIGURATION = {key: str(p.relative_to(ROOT)) for key, p in resolved.items()}
    RUN_CONFIGURATION['source_aliases'] = dict(aliases)
    SCOPE, SOURCE, ALIASES = scope, resolved['source'], dict(aliases)
    return dict(RUN_CONFIGURATION)


def workflow_binding():
    names = ('study_io.py', 'training_trace.py', 'configured_sandbox.py',
             'composition_study.py', 'composition_continuation.py',
             'composition_retention_audit.py', 'composition_final_audit.py')
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def read(path):
    return json.loads(path.read_text())


def new_text(path, value):
    exact_child(ROOT, path)
    with path.open('x') as stream:
        stream.write(value)


def samples_for(config, transform=None):
    fields = exact_child(ROOT, ROOT / config['fields'])
    if file_hash(fields / 'dataset.json') != config['fields_manifest_sha256']:
        raise ValueError('Input manifest changed')
    _, _, samples = load_study(fields, root=ROOT)
    order = [r['case'] for r in read(SCOPE / 'parent_replay_verified_v2.json')['rows']
             if r['mode'] == 'original']
    by_key = {s.key: s for s in samples}
    if set(order) != set(by_key):
        raise ValueError('Cohort changed')
    samples = [by_key[key] for key in order]
    if transform:
        move = lambda a: np.ascontiguousarray(transform_array(a, transform))
        samples = [replace(s, key=transform + '-' + s.key,
                           fields={k: move(v) for k, v in s.fields.items()},
                           human=move(s.human), known=move(s.known)) for s in samples]
    return samples


def protocol_for(config, seed):
    values = dict(config['evaluation'])
    for k in ('train_groups', 'validation_groups', 'audit_groups'):
        values[k] = tuple(values[k])
    values['seeds'] = (seed,)
    return Protocol(**values)


def measure(folder, samples, protocol, branch='composed'):
    rows = []
    for index, sample in enumerate(samples):
        with np.load(folder / f'prediction-{index:04d}.npz', allow_pickle=False) as z:
            metrics = evaluate(sample.validation_target(), dict(z), protocol)
        rows.append({'case': sample.key, 'group': sample.group, 'seed': protocol.seeds[0],
                     'role': protocol.split(sample.group), 'branch': branch, **metrics})
    return rows


def prepare():
    if RUN_CONFIGURATION is None:
        raise ValueError('Explicit --run-config required; no historical run is selected by default')
    exact_child(ROOT, SCOPE)
    receipt = read(SCOPE / 'parent_replay_verified_v2.json')
    if receipt['status'] != 'verified_parent_ready_for_new_study':
        raise ValueError('Actually verified parent required')
    source_protocol = read(SOURCE / 'protocol.json')
    diagnostic_path = ROOT / RUN_CONFIGURATION['source_diagnostics']
    old_request = read(diagnostic_path)['context']
    config = {k: source_protocol[k] for k in ('evaluation', 'fields', 'fields_manifest_sha256',
                                            'development_transforms', 'held_back_transforms')}
    config.update(format='whole-program-composition-study-v1', seeds=[23, 41, 67],
        arms=['full', 'route'], rounds_per_arm=12, model='gpt-5.6-sol', reasoning_effort='xhigh',
        new_blind_dataset=False, source_protocol_sha256=file_hash(SOURCE / 'protocol.json'),
        source_diagnostics_sha256=file_hash(diagnostic_path),
        run_configuration=RUN_CONFIGURATION, workflow_binding=workflow_binding(),
        parent_verification_sha256=file_hash(SCOPE / 'parent_replay_verified_v2.json'),
        engine_fingerprint=engine_fingerprint(),
        api_sha256=file_hash(Path(__file__).with_name('composition_runtime.py')),
        maximum_checkpoint_bytes=32*1024**2, maximum_prediction_pool_bytes=512*1024**2,
        cumulative_growth_bytes=1024**3, minimum_free_bytes=8*1024**3,
        success={'independent_full_trajectories': 2, 'accepted_updates_per_successful_trajectory': 2,
                 'minimum_nontraining_gain_over_frozen_and_route': .002,
                 'minimum_new_capacity_over_old_casewise_ceiling': .002,
                 'later_actual_use_of_accepted_state': True,
                 'all_final_transforms_after_search_sealed': True},
        previous_goal_turn='no RSI algorithm progress; completed read-only cleanup verification')
    if file_hash(SCOPE / 'parent_replay_verified_v2.json') != config['parent_verification_sha256']:
        raise ValueError('Parent changed')
    p = protocol_for(config, 41)
    samples = samples_for(config)
    original = measure(SCOPE / 'original_output', samples, p, 'frozen')
    rotated = measure(SCOPE / 'rot90_output', samples_for(config, 'rot90'), p, 'frozen')
    # Keep a conservative old-library ceiling, including rejected historical programs.
    gaps = old_request['bank_gap_diagnosis']
    if len(gaps) != len(original):
        raise ValueError('Old diagnostic cohort mismatch')
    ceilings = []
    for row, gap in zip(original, gaps):
        if (ALIASES[row['group']] != gap['group']
                or abs(row['coverage'] - gap['current_coverage']) > 1e-12):
            raise ValueError('Historical anonymous order/parent coverage mismatch')
        ceilings.append({'case': row['case'], 'group': row['group'],
                         'ceiling': max(row['coverage'], gap['best_existing_branch_coverage']),
                         'budget_upper_bound': gap['candidate_budget_coverage_upper_bound']})
    write_new(SCOPE / 'prospective_protocol.json', config)
    write_new(SCOPE / 'frozen_measurements.json', {'original': original, 'rot90': rotated,
              'ceiling': ceilings, 'old_diagnostic_record_sha256': config['source_diagnostics_sha256']})
    # Entire cohort and every prior recorded outcome; no selecting a favorable example.
    compact = {'observations': old_request['anonymous_observations'],
               'observable_feature_definitions': old_request['observable_feature_definitions'],
               'observable_information_by_source': {}, 'historical_outcomes': []}
    for group in ('D0', 'D1', 'D2'):
        records = [r for r in old_request['observable_field_information'] if r['group'] == group]
        compact['observable_information_by_source'][group] = records[:1]
    for outcome in old_request['all_attempt_feedback']:
        compact['historical_outcomes'].append({k: outcome.get(k) for k in
                                               ('action', 'status', 'decision', 'error')})
    write_new(SCOPE / 'compact_historical_evidence.json', compact)
    print(json.dumps({'prepared': True, 'cohort': len(samples), 'summary': summarize(original)}), flush=True)


def validate_study():
    if RUN_CONFIGURATION is None:
        raise ValueError('Explicit --run-config required; no historical run is selected by default')
    config = read(SCOPE / 'prospective_protocol.json')
    if (config.get('run_configuration') != RUN_CONFIGURATION
            or config.get('workflow_binding') != workflow_binding()):
        raise ValueError('Workflow/configuration changed; historical sealed runs use frozen commit 56fc212')
    if (engine_fingerprint() != config['engine_fingerprint']
            or file_hash(Path(__file__).with_name('composition_runtime.py')) != config['api_sha256']
            or file_hash(SCOPE / 'parent_replay_verified_v2.json') != config['parent_verification_sha256']):
        raise ValueError('Frozen evaluation/API/parent verification changed')
    manifest = read(SCOPE / 'frozen_parent/manifest.json')
    for name, expected in manifest['exported_hashes'].items():
        if file_hash(SCOPE / 'frozen_parent/program' / name) != expected:
            raise ValueError('Frozen program changed: ' + name)
    verify_snapshot(ROOT, read(SCOPE / 'protected_before.json'))
    return config


def history(arm, seed):
    directory = SCOPE / f'{arm}-s{seed}'
    records = []
    if directory.exists():
        for folder in sorted(directory.glob('r[0-9][0-9]')):
            if (folder / 'result.json').exists():
                records.append((folder, verified_result(folder)))
            else:
                raise RuntimeError('Previous attempt is not terminal: ' + str(folder))
    return records


def verified_result(folder):
    seal=read(folder/'result_binding.json')
    for name in ('result.json','response.json','request_binding.json'):
        if file_hash(folder/name)!=seal[name]:
            raise ValueError('Completed attempt evidence changed: '+name)
    return read(folder/'result.json')


INSTRUCTIONS = '''You are an independent scientific program proposer. Use only this packet;
do not read repository files, images, annotations, conversation history, or other trajectories.
The task is to create genuinely new microscopy interface-analysis capacity and integrate it.
Large changes/full algorithm redesign/composition are allowed. No required minor-edit direction.
You autonomously choose whether to train, change analysis, change candidate support, or compose.
Return ONE JSON object with observation, mechanism, action, applicability, falsifier,
training_steps (0..4096), pipeline_code. No Markdown fences. Do not run experiments yourself.
Full arm: implement predict(sample,state,context)->{'score':finite 2-D float,'support':2-D bool}.
Route control: implement route(sample,state,context)->{'branch':name,'reason':string}; allowed
names are frozen, parent, or available branch names. Route control cannot modify their outputs.
Optional fit(samples,context)->dict of finite numeric arrays when training_steps>0.
Only D0 labels enter fit. Unknown pixels are not negative labels. Inference NEVER sees labels,
case identity, source names, absolute filenames, evaluation code, or other images. Do not infer
case identity using checksums, hardcoded shapes/indexes, label coordinates, or case lookup tables.
Applicability must use structural/observable quantities and be executable in your program.
Fields: image, base, support, valid, physics, student, physics_valid, variation, domain_labels;
all arrays align with image. valid is analyzability, NOT old teacher support. Predict support
must be a subset of valid; it may expand beyond old support. No image size-specific rules.
Returned scores are normalized to float32 by the frozen runtime; numerical tie-breaking
must survive float32 conversion. This clarifies existing behavior, not an evaluator change.
context has seed, device ('cpu' or 'cuda'), training_steps. Optional training state is supplied
as state to predict; inherited weights can be used via parent calls. Runtime numpy/scipy/skimage/
torch are installed, no downloads. Numeric checkpoint <=32MiB; per execution <=900sec.
Import capabilities; capabilities.parent(sample,context) executes the accepted parent of YOUR
node (frozen on round1); capabilities.frozen_best(sample,context) executes unchanged F41-r06;
capabilities.branch(name,sample,context) executes an unmodified available branch. Functions
return score/support. Use these to compose or provide conditional fallback, NOT only tiny blends.
If calling capabilities from fit, pass only the nine observable fields, not human/known.
Branch output conventions are in frozen_code. No host filesystem/network/subprocess/evaluator IO.
The candidate budget is ceil(0.10*H*W) highest positive scores inside support. Metric is coverage
of expert-positive area, NOT IoU/accuracy/precision. Unknown pixels remain unknown.
Whole-system and nontraining D1,D2 mean improvement >=0.002 versus CURRENT accepted parent;
no source drops >0.002 versus frozen, current, or best retained score. rot90 must preserve its
own original and historical rotated source scores within0.002. Frozen library ceiling is an
OFFLINE diagnostic; never a deployable oracle or a source-lookup selector. Final hidden transforms
are not available during search. Historical data have been seen; do not call them fresh blind data.
Prefer a specific mechanistic hypothesis with a discriminating measurable failure condition.
Preserve all feedback; a proposed method is not successful until actually executed and measured.
'''


def request(arm, seed):
    from .composition_final_audit import require_search_open
    from .composition_continuation import budget_binding
    require_search_open(SCOPE)
    config = validate_study()
    if arm not in config['arms'] or seed not in config['seeds']:
        raise ValueError('Undeclared arm or seed')
    previous = history(arm, seed)
    number = len(previous) + 1
    continuation = budget_binding(config, arm, seed, number)
    accepted = [(folder, r) for folder, r in previous if r['decision']['accepted']]
    parent = accepted[-1][1]['identity'] if accepted else 'frozen'
    if continuation is not None and continuation['parent'] != parent:
        raise ValueError('Continuation parent is not the current admitted state')
    fixed = read(SCOPE / 'frozen_measurements.json')
    current = accepted[-1][1]['original']['rows'] if accepted else fixed['original']
    public = read(SCOPE / 'compact_historical_evidence.json')
    public.update(arm=arm, seed=seed, round=number, parent_node=parent,
                  current_summary={ALIASES[g]: v['mean'] for g,v in summarize(current).items()},
                  ceiling=[{'sample': i, 'group': ALIASES[r['group']],
                            'best_old_branch': r['ceiling'], 'budget_upper_bound': r['budget_upper_bound']}
                           for i,r in enumerate(fixed['ceiling'])],
                  current_cases=[{'sample':i,'group':ALIASES[r['group']], 'coverage':r['coverage'],
                                  'budget':r['budget'],'positive_pixels':r['denominator']}
                                 for i,r in enumerate(current)],
                  frozen_code={p.name:p.read_text() for p in (SCOPE/'frozen_parent/program').glob('*.py')
                               if p.name!='pipeline.py'},
                  prior_attempts=[{'proposal':read(f/'response.json'),
                                    'decision':r['decision'], 'summary':r.get('summary'),
                                    'new_capacity':r.get('new_capacity'),
                                    'case_change_diagnostics':read(f/'diagnostics_v1.json')
                                        if (f/'diagnostics_v1.json').exists() else None,
                                    'execution_error':r.get('error')}
                                   for f,r in previous],
                  accepted_parent_code=read(accepted[-1][0]/'response.json')['pipeline_code'] if accepted else None)
    if continuation is not None:
        public['continuation_phase'] = dict(continuation,
            note='One separately preregistered additional research call; unchanged scientific protocol. No outcome is prescribed.')
    folder = SCOPE / f'{arm}-s{seed}' / f'r{number:02d}'
    exact_child(ROOT, folder).mkdir(parents=True, exist_ok=False)
    packet = {'model':config['model'],'reasoning_effort':config['reasoning_effort'],
              'instructions':INSTRUCTIONS,'evidence':public,
              'response_schema':{'observation':'str','mechanism':'str','action':'str',
                   'applicability':'str','falsifier':'str','training_steps':'integer 0..4096',
                   'pipeline_code':'str'}}
    write_new(folder/'request.json',packet)
    write_new(folder/'request_binding.json',{'request_sha256':file_hash(folder/'request.json'),
        'protocol_sha256':file_hash(SCOPE/'prospective_protocol.json'), 'parent':parent,
        'seed':seed,'arm':arm,'round':number,'requested_model':config['model'],
        'requested_reasoning_effort':'xhigh','independent_context':True,
        'provider_backend_attestation':None,'provider_token_usage':None,
        **({'continuation_phase':continuation} if continuation is not None else {})})
    print(json.dumps({'request':str(folder/'request.json'),'bytes':(folder/'request.json').stat().st_size}),flush=True)


def checked_proposal(folder, config):
    from .composition_continuation import check_binding
    binding = read(folder / 'request_binding.json')
    if (file_hash(folder / 'request.json') != binding['request_sha256']
            or file_hash(SCOPE / 'prospective_protocol.json') != binding['protocol_sha256']):
        raise ValueError('Sealed request or protocol changed')
    check_binding(config, binding)
    proposal = read(folder / 'response.json')
    keys = {'observation', 'mechanism', 'action', 'applicability', 'falsifier',
            'training_steps', 'pipeline_code'}
    if set(proposal) != keys or any(not isinstance(proposal[k],str) or not proposal[k].strip()
                                  for k in keys - {'training_steps'}):
        raise ValueError('A complete single executable model proposal is required')
    steps = proposal['training_steps']
    if type(steps) is not int or not 0 <= steps <= config['evaluation']['maximum_training_steps']:
        raise ValueError('Training budget violation')
    tree = ast.parse(proposal['pipeline_code'])
    functions = {node.name for node in tree.body if isinstance(node,ast.FunctionDef)}
    required = {'route' if binding['arm']=='route' else 'predict'} | ({'fit'} if steps else set())
    if not required <= functions:
        raise ValueError('Required entry functions absent')
    # OS confinement remains the actual security boundary; these are contract checks.
    allowed = {'numpy','scipy','skimage','torch','math','random','collections','functools',
               'itertools','statistics','capabilities'}
    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node,ast.Import) else
                 [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
        if any(name.split('.')[0] not in allowed for name in names):
            raise ValueError('Undeclared module in scientific program')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {
                'open','exec','eval','compile','__import__'}:
            raise ValueError('Dynamic code/file IO is outside the proposal contract')
    return binding, proposal


def completed_before(arm, seed, number):
    records = []
    for index in range(1,number):
        folder=SCOPE/f'{arm}-s{seed}'/f'r{index:02d}'
        if not (folder/'result.json').is_file():
            raise ValueError('Prior attempt must be terminal before proceeding')
        records.append((folder,verified_result(folder)))
    return records


def package(folder, binding, proposal, prior):
    identity = digest({'proposal':proposal,'parent':binding['parent'],
                       'seed':binding['seed'],'arm':binding['arm']})
    program = folder/'program'
    exact_child(ROOT,program).mkdir(exist_ok=False)
    frozen = SCOPE/'frozen_parent/program'
    for name, expected in read(SCOPE/'frozen_parent/manifest.json')['exported_hashes'].items():
        source=frozen/name
        if file_hash(source)!=expected:
            raise ValueError('Frozen dependency changed')
        target = program/('retained.py' if name=='pipeline.py' else name)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    nodes = {}
    if binding['parent'] != 'frozen':
        parent_folder,parent_result=next((f,r) for f,r in prior if r['identity']==binding['parent'])
        if not parent_result['decision']['accepted']:
            raise ValueError('Rejected state cannot become parent')
        for name,expected in parent_result['program_hashes'].items():
            if file_hash(parent_folder/'program'/name)!=expected:
                raise ValueError('Inherited program changed')
        parent_config=read(parent_folder/'program/composition.json')
        nodes.update(parent_config['nodes'])
        for item in nodes.values():
            for name in (item['program'],item['checkpoint']):
                if name:
                    source=exact_child(parent_folder/'program',parent_folder/'program'/name)
                    with (program/name).open('xb') as stream:
                        stream.write(source.read_bytes())
        if parent_result['state']:
            state=exact_child(ROOT,ROOT/parent_result['state']['path'])
            if file_hash(state)!=parent_result['state']['sha256']:
                raise ValueError('Inherited trained state changed')
            name='weights_'+binding['parent']+'.npz'
            with (program/name).open('xb') as stream:
                stream.write(state.read_bytes())
            nodes[binding['parent']]['checkpoint']=name
    code='node_'+identity+'.py'
    new_text(program/code,proposal['pipeline_code'])
    nodes[identity]={'program':code,'checkpoint':None,'parent':binding['parent'],'arm':binding['arm']}
    write_new(program/'composition.json',{'current':identity,'nodes':nodes})
    new_text(program/'capabilities.py',Path(__file__).with_name('composition_runtime.py').read_text())
    new_text(program/'pipeline.py',ENTRY)
    return identity,program


def stage(folder, samples, pool, protocol, state=None, fit=False, steps=0, device='cpu'):
    exact_child(ROOT,folder).mkdir(exist_ok=False)
    records=[]
    for index,sample in enumerate(samples):
        if any(k not in FIELDS for k in sample.fields):
            raise ValueError('Unexpected observable field')
        fields=dict(sample.fields)
        if fit:
            if protocol.split(sample.group)!='train':
                raise ValueError('Weight-fit excluded source entered training')
            fields.update(human=sample.human,known=sample.known)
        records.append(pool.stage(folder,f'sample-{index:04d}.fields.json',fields))
    if state:
        source=exact_child(ROOT,state)
        os.link(source,folder/'state.npz')
    write_new(folder/'request.json',{'mode':'fit' if fit else 'predict','seed':protocol.seeds[0],
              'training_steps':steps,'samples':records,'state_file':bool(state),'device':device})


def execute(arm, seed, number, gpu=None):
    from .composition_final_audit import require_search_open
    require_search_open(SCOPE)
    config=validate_study()
    folder=exact_child(ROOT,SCOPE/f'{arm}-s{seed}'/f'r{number:02d}')
    if (folder/'execution_started.json').exists() or (folder/'result.json').exists():
        raise FileExistsError('Attempt already started; inspect receipts, never blindly restart')
    binding,proposal=checked_proposal(folder,config)
    if (binding['arm'],binding['seed'],binding['round'])!=(arm,seed,number):
        raise ValueError('Request identity mismatch')
    prior=completed_before(arm,seed,number)
    accepted=[(f,r) for f,r in prior if r['decision']['accepted']]
    parent=accepted[-1][1]['identity'] if accepted else 'frozen'
    if binding['parent']!=parent:
        raise ValueError('Request was not based on the actual last accepted state')
    write_new(folder/'execution_started.json',{'response_sha256':file_hash(folder/'response.json'),
        'controller_sha256':file_hash(Path(__file__)),'api_sha256':config['api_sha256'],
        'request_binding_sha256':file_hash(folder/'request_binding.json'),'gpu':gpu,
        'requested_model':config['model'],'requested_reasoning_effort':'xhigh',
        'backend_attestation':None,'token_usage':None})
    result={'identity':None,'parent':parent,'decision':{'accepted':False},'state':None,
            'status':'execution_failed','new_blind_dataset':False,'production_promoted':False}
    try:
        identity,program=package(folder,binding,proposal,prior)
        result['identity']=identity
        program_hashes={p.name:file_hash(p) for p in program.iterdir()}
        protocol=protocol_for(config,seed)
        samples=samples_for(config)
        pool=ArrayPool(ROOT,SCOPE/'field_pool',max_pool_bytes=256*1024**2,minimum_free_bytes=8*1024**3)
        prediction_pool=PredictionPool(ROOT,SCOPE/'prediction_pool',
            max_pool_bytes=config['maximum_prediction_pool_bytes'],minimum_free_bytes=8*1024**3)
        sandbox=SharedPredictionSandbox(ROOT,ROOT/config['run_configuration']['runtime'],timeout=900,gpu=gpu,
            prediction_pool=prediction_pool,max_written_prediction_bytes=64*1024**2)
        state=None
        device='cuda' if gpu is not None else 'cpu'
        if proposal['training_steps']:
            train=[s for s in samples if protocol.split(s.group)=='train']
            stage(folder/'fit_input',train,pool,protocol,fit=True,steps=proposal['training_steps'],device=device)
            training=sandbox.run(program,folder/'fit_input',folder/'fit_output')
            result['training']={'execution':training,'groups':list(protocol.train_groups),
                                'cases':[s.key for s in train],'requested_steps':proposal['training_steps']}
            if training['status']!='ok':
                raise RuntimeError((folder/'fit_output/stderr.log').read_text()[-4000:])
            state=folder/'fit_output/state.npz'
            if state.stat().st_size>config['maximum_checkpoint_bytes']:
                raise ValueError('Checkpoint storage contract exceeded')
            result['state']={'path':str(state.relative_to(ROOT)),'sha256':file_hash(state)}
        for mode in ('original','rot90'):
            cohort=samples if mode=='original' else samples_for(config,mode)
            inputs=folder/(mode+'_input');output=folder/(mode+'_output')
            stage(inputs,cohort,pool,protocol,state=state,device=device)
            print(json.dumps({'phase':mode,'arm':arm,'seed':seed,'round':number}),flush=True)
            execution=sandbox.run(program,inputs,output)
            if execution['status']!='ok':
                raise RuntimeError((output/'stderr.log').read_text()[-4000:])
            rows=measure(output,cohort,protocol)
            result[mode]={'rows':rows,'summary':summarize(rows),'execution':execution,
                          'actual_execution':True,'transform':mode,'router_training':None}
        fixed=read(SCOPE/'frozen_measurements.json')
        normalize=lambda rows:[dict(r,seed=seed) for r in rows]
        frozen=normalize(fixed['original'])
        previous=normalize(accepted[-1][1]['original']['rows']) if accepted else frozen
        best={g:v['mean'] for g,v in summarize(frozen).items()}
        rotated_best={g:v['mean'] for g,v in summarize(fixed['rot90']).items()}
        for _,record in accepted:
            for g,v in record['original']['summary'].items():best[g]=max(best[g],v['mean'])
            for g,v in record['rot90']['summary'].items():rotated_best[g]=max(rotated_best[g],v['mean'])
        quality=expansion_admission(result['original']['rows'],previous,frozen,protocol,best)
        result['decision']=stability_admission(quality,result['original']['rows'],
            {'rot90':result['rot90']},('rot90',),protocol.maximum_group_regression,{'rot90':rotated_best})
        ceilings={r['case']:r for r in fixed['ceiling']}
        new_gain={g:float(np.mean([r['coverage']-ceilings[r['case']]['ceiling']
                                    for r in result['original']['rows'] if r['group']==g])) for g in best}
        result['new_capacity']={'gain_over_conservative_old_casewise_ceiling':new_gain,
            'nontraining_mean':float(np.mean([new_gain[g] for g in protocol.validation_groups])),
            'oracle_used_at_inference':False,'not_a_separate_deployable_baseline':True}
        # Novelty is reported independently; a useful non-novel upgrade may be retained,
        # but cannot satisfy the goal's new-capacity completion criterion.
        result['novel_capacity_verified']=result['new_capacity']['nontraining_mean']>=protocol.minimum_gain
        result['summary']=result['original']['summary']
        if result['decision']['accepted']:
            # Real later-order execution is required before admission, not cached reranking.
            stage(folder/'reverse_input',list(reversed(samples)),pool,protocol,state=state,device=device)
            execution=sandbox.run(program,folder/'reverse_input',folder/'reverse_output')
            if execution['status']!='ok':raise RuntimeError('Accepted candidate reverse execution failed')
            for i in range(len(samples)):
                a=folder/'original_output'/f'prediction-{i:04d}.npz'
                b=folder/'reverse_output'/f'prediction-{len(samples)-1-i:04d}.npz'
                with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
                    if any(not np.array_equal(left[k],right[k]) for k in ('score','support')):
                        raise ValueError('Inference depends on sample order')
            result['reverse_replay']={'execution':execution,'all_arrays_exact':True,'cases':len(samples)}
        if any(file_hash(program/name)!=h for name,h in program_hashes.items()):
            raise ValueError('Readonly executable package changed')
        if state and file_hash(state)!=result['state']['sha256']:
            raise ValueError('Fitted state changed during prediction')
        result['program_hashes']=program_hashes
        result['protected_check']=verify_snapshot(ROOT,read(SCOPE/'protected_before.json'))
        result['status']='accepted' if result['decision']['accepted'] else 'measured_rejected'
    except (ValueError,RuntimeError,KeyError,OSError) as error:
        result['error']=str(error)
        result['decision']={'accepted':False,'reason':'execution_or_integrity_failure'}
    result['response_sha256']=file_hash(folder/'response.json')
    if result['response_sha256']!=read(folder/'execution_started.json')['response_sha256']:
        raise ValueError('Model response changed during execution')
    write_new(folder/'result.json',result)
    write_new(folder/'result_binding.json',{name:file_hash(folder/name) for name in
              ('result.json','response.json','request_binding.json')})
    print(json.dumps({k:result.get(k) for k in ('status','identity','decision','new_capacity','summary','error')}),flush=True)


def error_signal_profiles(fields, human, selected):
    """Label-side distributions, never labels/coordinates sent to inference.

    Unknown pixels are only the observable rank reference, never negatives.
    These are development diagnostics, not new held-out evaluation evidence.
    """
    valid=np.asarray(fields['valid'],bool)
    initial=np.asarray(fields['support'],bool)
    physics_valid=np.asarray(fields['physics_valid'],bool)
    positive=np.asarray(human,bool)
    selected=np.asarray(selected,bool)
    masks={
        'selected_known_positive':positive & selected & valid,
        'missed_known_positive_inside_initial_support':positive & ~selected & initial & valid,
        'missed_known_positive_outside_initial_support_physics_valid':
            positive & ~selected & ~initial & physics_valid & valid,
        'missed_known_positive_outside_physics_valid_but_valid':
            positive & ~selected & ~physics_valid & valid,
    }
    output={'label_side_development_diagnostic_only':True,
            'unknown_pixels_are_not_negatives':True,
            'rank_reference':'all finite observable pixels inside valid; average ties',
            'known_positive_outside_valid':int((positive & ~valid).sum()),
            'regions':{name:{'positive_pixels':int(mask.sum()),'fields':{}}
                       for name,mask in masks.items()}}
    for name in ('image','base','physics','student','variation'):
        if name not in fields:continue
        values=np.asarray(fields[name],dtype=np.float64)
        finite=np.isfinite(values)
        reference=np.sort(values[valid & finite])
        if not reference.size:continue
        for region,mask in masks.items():
            subset=values[mask & finite]
            if not subset.size:continue
            left=np.searchsorted(reference,subset,side='left')
            right=np.searchsorted(reference,subset,side='right')
            ranks=(left+right-1)/(2*max(1,reference.size-1))
            output['regions'][region]['fields'][name]={
                'raw_q10_q50_q90':np.quantile(subset,[.1,.5,.9]).tolist(),
                'valid_midrank_q10_q50_q90':np.quantile(ranks,[.1,.5,.9]).tolist(),
            }
    return output


def domain_signal_profile(fields):
    """Observable-only field availability; no expert labels or coordinates."""
    labels=np.asarray(fields['domain_labels'])
    output={'observable_only':True,'regions':{}}
    for name in ('valid','support','physics_valid'):
        mask=np.asarray(fields[name],bool) & np.isfinite(labels)
        values,counts=np.unique(labels[mask],return_counts=True)
        horizontal=mask[:,1:] & mask[:,:-1]
        vertical=mask[1:,:] & mask[:-1,:]
        pairs=int(horizontal.sum()+vertical.sum())
        transitions=int((horizontal & (labels[:,1:]!=labels[:,:-1])).sum()
                        +(vertical & (labels[1:,:]!=labels[:-1,:])).sum())
        output['regions'][name]={'pixels':int(mask.sum()),'distinct_values':int(values.size),
            'value_counts':[[float(v),int(n)] for v,n in zip(values,counts)] if values.size<=16 else None,
            'internal_adjacent_pairs':pairs,'internal_domain_transitions':transitions,
            'transition_fraction':transitions/max(1,pairs)}
    return output


def training_supervision_summary(samples, protocol):
    """Anonymous counts from permitted fit sources, never invented negatives."""
    sources = {}
    for sample in samples:
        if protocol.split(sample.group) != 'train':
            continue
        valid = np.asarray(sample.fields['valid'], bool)
        human = np.asarray(sample.human, bool)
        weight = np.asarray(sample.known)
        if human.shape != valid.shape or weight.shape != valid.shape:
            raise ValueError('Training supervision shape mismatch')
        known = valid & np.isfinite(weight) & (weight > 0)
        row = sources.setdefault(ALIASES.get(sample.group, sample.group), {
            'cases': 0, 'valid_pixels': 0, 'known_positive_pixels': 0,
            'known_negative_pixels': 0, 'unknown_pixels': 0,
            'positive_without_known_weight_pixels': 0})
        row['cases'] += 1
        row['valid_pixels'] += int(valid.sum())
        row['known_positive_pixels'] += int((known & human).sum())
        row['known_negative_pixels'] += int((known & ~human).sum())
        row['unknown_pixels'] += int((valid & ~known).sum())
        row['positive_without_known_weight_pixels'] += int((valid & human & ~known).sum())
    return {'fit_sources_only': True, 'unknown_pixels_are_not_negatives': True,
            'counts_are_available_before_candidate_subsampling': True, 'sources': sources}


def fitted_state_summary(result):
    """Inspect small state metadata without sending weights or claiming convergence."""
    state = result.get('state')
    if state is None:
        return None
    path = exact_child(ROOT, ROOT / state['path'])
    if file_hash(path) != state['sha256']:
        raise ValueError('Fitted state changed before diagnosis')
    with np.load(path, allow_pickle=False) as arrays:
        scalar = {key: arrays[key].reshape(-1)[0].item() for key in
                  ('trained', 'positive_count', 'negative_count', 'has_negative', 'audit_steps')
                  if key in arrays and arrays[key].size == 1}
        shapes = {key: list(arrays[key].shape) for key in arrays.files}
        # A fitted "do not intervene" decision is useful feedback too. Only
        # bounded named diagnostics are exposed, never arbitrary model weights.
        decisions = {}
        for key in ('full_quota_fraction', 'fit_mean_gain', 'fit_samples_used',
                    'route_index', 'route_gain', 'route_count'):
            if key not in arrays:
                continue
            value = arrays[key]
            if (value.dtype.kind in 'biuf' and 0 < value.size <= 8
                    and np.all(np.isfinite(value))):
                decisions[key] = value.reshape(-1).tolist()
    return {'checkpoint_bytes': path.stat().st_size, 'state_shapes': shapes,
            'candidate_reported_scalar_metadata': scalar,
            'candidate_reported_training_decisions': decisions,
            'metadata_is_not_independent_proof_of_iterations_or_training_gain': True}


def diagnose(arm,seed,number):
    config=validate_study()
    folder=exact_child(ROOT,SCOPE/f'{arm}-s{seed}'/f'r{number:02d}')
    result=verified_result(folder)
    if 'original' not in result:raise ValueError('No successful original prediction to diagnose')
    previous=completed_before(arm,seed,number)
    if result['parent']=='frozen':parent_folder=SCOPE
    else:parent_folder=next(f for f,r in previous if r['identity']==result['parent'])
    samples=samples_for(config)
    records=[]
    for index,sample in enumerate(samples):
        def load(directory):
            with np.load(directory/'original_output'/f'prediction-{index:04d}.npz',allow_pickle=False) as z:
                return dict(z)
        old,new=load(parent_folder),load(folder)
        budget=result['original']['rows'][index]['budget']
        def selected(prediction):
            indices=np.flatnonzero(prediction['support'] & (prediction['score']>0))
            ids=indices[np.argsort(-prediction['score'].ravel()[indices],kind='stable')[:budget]]
            mask=np.zeros(sample.human.shape,bool);mask.ravel()[ids]=True
            return mask
        old_mask,new_mask=selected(old),selected(new)
        gained=new_mask & ~old_mask;lost=old_mask & ~new_mask
        trace=read(folder/'original_output'/f'composition-trace-{index:04d}.json')['calls']
        region=(sample.fields['valid'].astype(bool) & sample.fields['physics_valid'].astype(bool)
                & ~sample.fields['support'].astype(bool))
        field_information={}
        for name in ('base','physics','variation','student'):
            values=np.asarray(sample.fields[name])[region]
            if values.size:
                field_information[name]={'pixels':int(values.size),'zero_fraction':float(np.mean(values==0)),
                    'std':float(np.std(values)),'min':float(values.min()),'max':float(values.max()),
                    'q10_q50_q90':[float(v) for v in np.quantile(values,[.1,.5,.9])]}
        records.append({'sample':index,'group':ALIASES[sample.group],
            'initial_support_fraction':float(np.mean(sample.fields['support'])),
            'physics_valid_fraction':float(np.mean(sample.fields['physics_valid'])),
            'old_support_pixels':int(old['support'].sum()),'new_support_pixels':int(new['support'].sum()),
            'changed_score_pixels':int(np.count_nonzero(old['score']!=new['score'])),
            'added_selected_pixels':int(gained.sum()),'removed_selected_pixels':int(lost.sum()),
            'added_known_positive_pixels':int((gained & sample.human).sum()),
            'removed_known_positive_pixels':int((lost & sample.human).sum()),
            'observable_fields_outside_initial_support_within_physics_valid':field_information,
            'observable_domain_field_availability':domain_signal_profile(sample.fields),
            'parent_error_signal_profiles':error_signal_profiles(sample.fields,sample.human,old_mask),
            'calls':dict(Counter(x['call']+':'+str(x.get('branch',x.get('node',''))) for x in trace))})
    value={'result_sha256':file_hash(folder/'result.json'),
           'diagnostic_only_not_inference_data':True,'unknown_pixels_are_not_negatives':True,
           'prediction_computed_and_stored':True,'cases':records,
           'supervision_available_for_fit':training_supervision_summary(samples,protocol_for(config,seed)),
           'fitted_state_metadata':fitted_state_summary(result),
           'cases_with_changed_selection':sum(r['added_selected_pixels']+r['removed_selected_pixels']>0 for r in records)}
    write_new(folder/'diagnostics_v1.json',value)
    print(json.dumps({'diagnosed':str(folder),'cases_with_changed_selection':value['cases_with_changed_selection']}),flush=True)


def report():
    from .storage_budget import usage
    config=validate_study()
    trajectories=[]
    inference_count=0
    for arm in config['arms']:
        for seed in config['seeds']:
            records=[]
            directory=SCOPE/f'{arm}-s{seed}'
            for folder in sorted(directory.glob('r[0-9][0-9]')):
                record={'round':folder.name,'path':str(folder.relative_to(ROOT)),
                        'phase':'initial' if int(folder.name[1:]) <= config['rounds_per_arm'] else 'registered_continuation'}
                if (folder/'result.json').exists():
                    result=verified_result(folder)
                    record.update({k:result.get(k) for k in
                        ('identity','parent','status','decision','new_capacity','summary','error')})
                    record['result_sha256']=file_hash(folder/'result.json')
                    record['training_steps_requested']=read(folder/'response.json')['training_steps']
                    for key in ('original','rot90'):
                        if key in result:inference_count+=len(result[key]['rows'])
                    inference_count+=result.get('reverse_replay',{}).get('cases',0)
                else:
                    record['status']=('started_without_terminal_receipt' if (folder/'execution_started.json').exists()
                        else 'response_ready' if (folder/'response.json').exists() else 'request_prepared')
                records.append(record)
            trajectories.append({'arm':arm,'seed':seed,'attempts':records,
                'initial_budget_rounds':sum(r['phase']=='initial' for r in records),
                'additional_registered_rounds':sum(r['phase']=='registered_continuation' for r in records),
                'accepted_updates':sum(bool(r.get('decision',{}).get('accepted')) for r in records)})
    destination=SCOPE/'progress_reports'
    destination.mkdir(exist_ok=True)
    path=destination/f'progress-{len(list(destination.glob("progress-*.json")))+1:04d}.json'
    value={'status':'active_not_complete','goal_fully_achieved':False,
           'trajectories':trajectories,'actual_candidate_inferences':inference_count,
           'criteria':config['success'],'final_transforms_evaluated':False,
           'production_promoted':False,'protected_check':verify_snapshot(ROOT,read(SCOPE/'protected_before.json')),
           'storage':usage(SCOPE),'source_protocol_sha256':file_hash(SCOPE/'prospective_protocol.json'),
           'not_a_completion_audit':True,'new_scientific_paths_deleted':0}
    write_new(path,value)
    print(json.dumps({'report':str(path),'inferences':inference_count,
         'accepted':[t['accepted_updates'] for t in trajectories],'storage':value['storage']}),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','request','execute','diagnose','report',
                                          'continue','retention','seal','audit'))
    parser.add_argument('--run-config',required=True,help='Explicit workspace-local JSON; see configs/rsi_composition.example.json')
    parser.add_argument('--arm',choices=('full','route'),default='full')
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args(argv)
    configure_run(args.run_config)
    if args.command=='prepare':prepare()
    elif args.command=='request':request(args.arm,args.seed)
    elif args.command=='execute':execute(args.arm,args.seed,args.round,args.gpu)
    elif args.command=='diagnose':diagnose(args.arm,args.seed,args.round)
    elif args.command=='report':report()
    elif args.command=='continue':
        from .composition_continuation import register
        register(args.arm,args.seed)
    elif args.command=='retention':
        from .composition_retention_audit import audit
        audit()
    else:
        from . import composition_final_audit
        (composition_final_audit.seal if args.command=='seal' else composition_final_audit.execute)()


if __name__=='__main__':
    main()
