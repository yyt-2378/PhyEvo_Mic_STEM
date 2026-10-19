"""Anonymous aggregate error evidence; never inspect the transfer-audit split.

Single-field probe algorithms are explicit reference candidates, not discoveries
credited to the proposer. Expert positives and unknown pixels remain distinct.
"""
import numpy as np
from scipy import ndimage as ndi

from .contracts import Proposal
from .controller import BASELINE_POLICY
from .evaluation import evaluate

FIELDS = ('base', 'physics', 'student', 'variation', 'image')


def probe_proposals():
    result=[]
    for field in FIELDS:
        for sign in (1, -1):
            code=("import numpy as np\ndef predict(sample,state,context):\n"
                  f"    value={sign}*np.asarray(sample[{field!r}],dtype=np.float64)\n"
                  "    valid=np.asarray(sample['valid'],bool)\n"
                  "    score=np.zeros(value.shape,np.float64)\n"
                  "    idx=np.flatnonzero(valid.ravel())\n"
                  "    order=np.argsort(value.ravel()[idx],kind='stable')\n"
                  "    score.ravel()[idx[order]]=np.arange(1,len(idx)+1)/max(1,len(idx))\n"
                  "    return {'score':score,'support':valid}\n")
            p=Proposal('generate',(),f'Prespecified single-field diagnostic: {field}, sign {sign}',
                'Measure the existing field before attributing any gain to research',
                'Rank this field inside physical validity, under the same fixed-area evaluator',
                'A reference score, not an autonomous discovery','Any contract violation',
                code,BASELINE_POLICY)
            result.append((f'{field}:{sign:+}',p))
    return result


def selected_mask(score,support,fraction):
    score=np.asarray(score);support=np.asarray(support,bool)
    n=max(1,int(np.ceil(fraction*score.size)))
    idx=np.flatnonzero(support.ravel() & (score.ravel()>0))
    chosen=idx[np.argsort(-score.ravel()[idx],kind='stable')[:n]]
    out=np.zeros(score.shape,bool);out.ravel()[chosen]=True
    return out


def label_shape(human):
    """Only shape distributions: no positions, polygons or reconstruction data."""
    human=np.asarray(human,bool)
    labels,n=ndi.label(human)
    sizes=np.bincount(labels.ravel())[1:]
    distance=ndi.distance_transform_edt(human)[human]
    return {'positive_fraction':float(human.mean()),'connected_components':int(n),
            'largest_component_fraction_of_positive':float(sizes.max()/max(1,human.sum())) if sizes.size else 0.,
            'positive_interior_distance_q25_q50_q90_px':np.quantile(distance,[.25,.5,.9]).tolist() if distance.size else [0.]*3}


def field_enrichment(values,human,valid):
    """Histograms are feedback, not confirmed-negative labels."""
    values=np.asarray(values,float);mask=np.asarray(valid,bool)&np.isfinite(values)
    all_values=values[mask];positive=values[mask & human]
    if not all_values.size or not positive.size:
        return {'positive_pixels':int(positive.size)}
    sorted_values=np.sort(all_values)
    ranks=np.searchsorted(sorted_values,positive,side='right')/len(sorted_values)
    return {'positive_pixels':int(positive.size),
            'positive_percentile_q10_q50_q90':np.quantile(ranks,[.1,.5,.9]).tolist(),
            'field_std':float(np.std(all_values))}


def boundary_diagnostic(fields,human):
    labels=np.asarray(fields.get('domain_labels',np.zeros(human.shape)),dtype=np.int64)
    valid=np.asarray(fields['valid'],bool)
    boundary=np.zeros(human.shape,bool)
    boundary[1:] |= (labels[1:]!=labels[:-1]) & (labels[1:]>0) & (labels[:-1]>0)
    boundary[:,1:] |= (labels[:,1:]!=labels[:,:-1]) & (labels[:,1:]>0) & (labels[:,:-1]>0)
    boundary &= valid
    total=max(1,int(human.sum()))
    distance=ndi.distance_transform_edt(~boundary) if boundary.any() else np.full(human.shape,1e9)
    cuts=(0,2,4,8,16,32,1e10)
    bins=[(distance>=a)&(distance<b) for a,b in zip(cuts[:-1],cuts[1:])]
    return {'domain_boundaries_present':float(boundary.any()),
            'positive_mass_in_distance_bins_0_2_4_8_16_32_inf_px':[float((human&b).sum()/total) for b in bins],
            'valid_area_in_same_distance_bins':[float((valid&b).sum()/max(1,valid.sum())) for b in bins]}


def diagnose(samples,protocol,predictions):
    rows={'train':[],'validation':[]}
    for sample in samples:
        split=protocol.split(sample.group)
        if split not in rows:continue
        human=np.asarray(sample.human,bool);valid=np.asarray(sample.fields['valid'],bool)
        total=max(1,int(human.sum()))
        score,support=predictions.get(sample.key,(sample.fields['base'],sample.fields['support']))
        chosen=selected_mask(score,support,protocol.area_fraction)
        record={'shape':list(human.shape),'positive_pixels':int(human.sum()),
            'explicit_negative_pixels':int(((np.asarray(sample.known)>0)&~human).sum()),
            'annotation_shape':label_shape(human),
            'positive_mass_outside_support':float((human&~support).sum()/total),
            'positive_mass_inside_support_but_not_selected':float((human&support&~chosen).sum()/total),
            'budget_only_coverage_upper_bound':min(1.,np.ceil(protocol.area_fraction*human.size)/total),
            'fields':{key:field_enrichment(sample.fields[key],human,valid) for key in FIELDS}}
        record['domain_boundary_geometry']=boundary_diagnostic(sample.fields,human)
        rb=selected_mask(sample.fields['physics'],valid,protocol.area_fraction)
        rs=selected_mask(sample.fields['student'],valid,protocol.area_fraction)
        record['positive_mass_by_prior_agreement']={
            'both':float((human&rb&rs).sum()/total),
            'physical_only':float((human&rb&~rs).sum()/total),
            'student_only':float((human&~rb&rs).sum()/total),
            'neither':float((human&~rb&~rs).sum()/total)}
        rows[split].append(record)
    def reduce(values):
        first=values[0]
        if isinstance(first,dict):return {k:reduce([v[k] for v in values]) for k in first}
        return np.mean(np.asarray(values,float),axis=0).tolist()
    return {split:{'planes':len(values),'mean_per_plane':reduce(values) if values else {},
                   'interpretation':'positive-unknown; means, not raw labels or coordinates; train uses frozen base, validation uses incumbent'}
            for split,values in rows.items()}


def validate_decision(proposal,visible_ids,protocol):
    proposal.validate(protocol)
    if proposal.target=='research_policy':raise ValueError('Task proposals cannot self-promote a research policy')
    if not set(proposal.parents)<=set(visible_ids):raise ValueError('Choose only visible parents')
    if proposal.training_steps and proposal.target!='learning':raise ValueError('Training must be declared as learning')
    if proposal.target=='learning' and not proposal.training_steps:raise ValueError('Learning direction requires actual training')
    return proposal


def available_actions(stagnation):
    # Eligibility has no round-modulo/remaining-budget conjunction.
    return {'allowed':['generate','tune','evolve','repair'],
            'consecutive_rounds_without_admission':stagnation,
            'instruction':'Choose the operator and parentage yourself. No action or branch is mandatory. '
            'If repeated fine-tuning stagnates, justify continuing it versus a different mechanism or learning. '
            'Generation and crossover remain eligible even in the last round.'}
