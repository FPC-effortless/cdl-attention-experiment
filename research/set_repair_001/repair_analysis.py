"""Predeclared paired outcome screen and descriptive latent-scale diagnostics."""
from pathlib import Path
import json,hashlib
import numpy as np
from experiment import ROOT,aggregate,ARMS,SEEDS,TEMPLATES

def analyze(root=ROOT):
    out=root/'results';records=[json.loads(x) for x in (out/'predictions.jsonl').open()];probes=[json.loads(x) for x in (out/'probes.jsonl').open()]
    vectors=np.load(out/'root_states.npz',allow_pickle=False)['states'];summary=json.loads((out/'summary.json').read_text())
    stages=('starting','control','invariance');cells=[];screens=[];paired=[]
    for arm in ARMS:
        for seed in SEEDS:
            lookups={}
            for stage in stages:
                normal=[r for r in records if (r['arm'],r['seed'],r['stage'],r['condition'])==(arm,seed,stage,'normal')]
                erased=[r for r in records if (r['arm'],r['seed'],r['stage'],r['condition'])==(arm,seed,stage,'erase_values')]
                lookups[stage]={(r['world_id'],r['query']):r for r in normal}
                g=[r for r in probes if (r['arm'],r['seed'],r['stage'])==(arm,seed,stage)]
                def small(a):return {'correct':sum(r['correct'] for r in a),'total':len(a),'accuracy':sum(r['correct'] for r in a)/len(a)}
                probe={}
                for p in ('baseline','same_value','changed_value'):
                    a=[r for r in g if (r['old'] is None if p=='baseline' else r['old']==r['new'] if p=='same_value' else r['old'] is not None and r['old']!=r['new'])];probe[p]=small(a)
                one={(r['pair_id'],r['world']):r for r in normal if r['query']=='r0'};pairids=sorted({k[0] for k in one})
                va=np.array([vectors[one[p,0]['trace_index']] for p in pairids]);vb=np.array([vectors[one[p,1]['trace_index']] for p in pairids]);means=(va+vb)/2
                within=float(np.mean((va-vb)**2));variance=float(np.mean((means-means.mean(0))**2));centroids=[]
                for new in range(8):centroids.append(means[[one[p,0]['new']==f'v{new}' for p in pairids]].mean(0))
                between=float(np.mean([np.mean((centroids[a]-centroids[b])**2) for a in range(8) for b in range(a)]))
                cells.append({'arm':arm,'seed':seed,'stage':stage,'overall':aggregate(normal),'erased':aggregate(erased),
                    'concrete':small([r for r in normal if r['answer']!=19]),'unknown':small([r for r in normal if r['answer']==19]),
                    'templates':{t:aggregate([r for r in normal if r['template']==t]) for t in TEMPLATES},
                    'roles':{role:small([r for r in normal if r['role']==role]) for role in ('root','alias','spare','unrelated')},
                    'probe':probe,'query_preserved':sum(r['query_preserved'] for r in normal),
                    'latent':{'paired_old_mse':within,'rms':float(np.sqrt(np.mean(np.concatenate([va,vb])**2))),'pairmean_variance':variance,'within_over_variance':within/(variance+1e-12),'between_new_centroid_mse':between}})
            for source,target in (('control','invariance'),('starting','control'),('starting','invariance')):
                a,b=lookups[source],lookups[target]
                if a.keys()!=b.keys():raise ValueError('unpaired evaluation')
                paired.append({'arm':arm,'seed':seed,'from':source,'to':target,'queries':len(a),
                    'rescued':sum(not a[k]['correct'] and b[k]['correct'] for k in a),'harmed':sum(a[k]['correct'] and not b[k]['correct'] for k in a)})
        a=[c for c in cells if c['arm']==arm and c['stage']=='control'];b=[c for c in cells if c['arm']==arm and c['stage']=='invariance']
        q=[y['overall']['accuracy']-x['overall']['accuracy'] for x,y in zip(a,b)];m=[y['overall']['map_accuracy']-x['overall']['map_accuracy'] for x,y in zip(a,b)]
        drops=[{'seed':x['seed'],'template':t,'delta':y['templates'][t]['accuracy']-x['templates'][t]['accuracy']} for x,y in zip(a,b) for t in TEMPLATES]
        same=[y['probe']['same_value']['correct']-x['probe']['same_value']['correct'] for x,y in zip(a,b)]
        criteria={'mean_query_improves':sum(q)>0,'mean_map_improves':sum(m)>0,'no_seed_query_regression':all(d>=0 for d in q),
            'no_template_seed_drop_over_2pp':all(d['delta']>=-.02-1e-12 for d in drops),'no_same_value_seed_regression':all(d>=0 for d in same),
            'normal_query_preservation_100pct':all(c['query_preserved']==c['overall']['total'] for c in b)}
        screens.append({'arm':arm,'criteria':criteria,'pass':all(criteria.values()),'query_deltas':q,'map_deltas':m,'template_deltas':drops,'same_value_count_deltas':same})
    report={'experiment':'SET-REPAIR-001','scope':'KNOWN_TEMPLATE_DEV_OBJECTIVE_INTERVENTION_NOT_ARCHITECTURE_QUALIFICATION','new_fits':12,'validation_pairs':summary['validation_pairs'],
        'cells':cells,'paired':paired,'repair_screens':screens,'predictions_sha256':hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest()}
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    r=analyze();print(json.dumps({'screens':r['repair_screens'],'validation_pairs':r['validation_pairs']},indent=2))
