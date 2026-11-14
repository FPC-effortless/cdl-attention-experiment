"""Predeclared deterministic summaries for the full correlated diagnostic grid."""
import itertools

def summarize(records):
    if len(records)!=13824:raise ValueError('incomplete factorial')
    cells=[];comparisons=[];all_lookups={}
    for arm,seed,heads,zero,full in itertools.product(('csm','pointer_slot'),(101,202,303),('normal','categorical'),(False,True),(False,True)):
        group=[r for r in records if (r['arm'],r['seed'],r['heads'],r['zero_reads'],r['full_gain'])==(arm,seed,heads,zero,full)]
        lookup={(r['key'],r['old'],r['new']):r for r in group}
        if len(group)!=288 or len(lookup)!=288:raise ValueError('duplicate/missing cell')
        all_lookups[arm,seed,heads,zero,full]=lookup
        stats={}
        for name,part in [('baseline',[r for r in group if r['old'] is None]),('diagonal',[r for r in group if r['old']==r['new']]),('off_diagonal',[r for r in group if r['old'] is not None and r['old']!=r['new']])]:
            stats[name]={'correct':sum(r['correct'] for r in part),'total':len(part),
                'by_new':{f'v{i}':sum(r['correct'] for r in part if r['new']==f'v{i}') for i in range(8)}}
        over=[r for r in group if r['old'] is not None];off=[r for r in over if r['old']!=r['new']]
        paired=[(lookup[r['key'],None,r['new']],r) for r in over]
        delta=max(abs(a-b) for x,y in paired for a,b in zip(x['logits'],y['logits']))
        old_range=max(max(lookup[f'r{k}',f'v{o}',f'v{n}']['logits'][j] for o in range(8))-min(lookup[f'r{k}',f'v{o}',f'v{n}']['logits'][j] for o in range(8)) for k in range(4) for n in range(8) for j in range(20))
        cells.append({'arm':arm,'seed':seed,'heads':heads,'zero_reads':zero,'full_gain':full,'accuracy':stats,
            'raw_head_error_cases':sum(any(r['raw_head_errors'].values()) for r in group),
            'query_preserved':sum(r['query_preserved'] for r in group),'untouched_set_preserved':sum(r['untouched_set_preserved'] for r in group),
            'target_retained_until_query':sum(r['target_retained_until_query'] for r in group),
            'overwrite_vs_baseline':{'changed':sum(x['prediction']!=y['prediction'] for x,y in paired),'harmed':sum(x['correct'] and not y['correct'] for x,y in paired),'rescued':sum(not x['correct'] and y['correct'] for x,y in paired),'maximum_absolute_logit_difference':delta,'minimum_top_two_margin':min(r['top_two_margin'] for r in group)},
            'maximum_old_conditioned_logit_range':old_range,
            'off_diagonal_old_answers':sum(r['prediction']==r['old'] for r in off),
            'newly_induced_old_answers':sum(r['prediction']==r['old'] and lookup[r['key'],None,r['new']]['prediction']!=r['old'] for r in off),
            'old_answers_already_baseline':sum(r['prediction']==r['old'] and lookup[r['key'],None,r['new']]['prediction']==r['old'] for r in off),
            'pointer_invariant_required':arm=='pointer_slot' and heads=='categorical' and zero and full,
            'pointer_invariant_pass':delta<=1e-5 and all(x['prediction']==y['prediction'] for x,y in paired) if arm=='pointer_slot' and heads=='categorical' and zero and full else None})
    for arm,seed,heads in itertools.product(('csm','pointer_slot'),(101,202,303),('normal','categorical')):
        for axis,fixed in itertools.product(('zero_reads','full_gain'),(False,True)):
            ca=(False,fixed) if axis=='zero_reads' else (fixed,False)
            cb=(True,fixed) if axis=='zero_reads' else (fixed,True)
            a=all_lookups[arm,seed,heads,*ca];b=all_lookups[arm,seed,heads,*cb]
            for panel in ('baseline','diagonal','off_diagonal'):
                keys=[k for k in a if (k[1] is None if panel=='baseline' else (k[1]==k[2] if panel=='diagonal' else k[1] is not None and k[1]!=k[2]))]
                comparisons.append({'arm':arm,'seed':seed,'heads':heads,'axis':axis,'other_axis_value':fixed,'panel':panel,'total':len(keys),
                    'changed':sum(a[k]['prediction']!=b[k]['prediction'] for k in keys),'rescued':sum(not a[k]['correct'] and b[k]['correct'] for k in keys),'harmed':sum(a[k]['correct'] and not b[k]['correct'] for k in keys)})
    return {'experiment':'OVERWRITE-PATHS-051','predictions':len(records),'model_fits':0,'cells':cells,'paired_interventions':comparisons,
        'scope':'FROZEN_POST_HOC_PATHWAY_DIAGNOSTIC_NOT_LEARNED_QUALIFICATION'}
