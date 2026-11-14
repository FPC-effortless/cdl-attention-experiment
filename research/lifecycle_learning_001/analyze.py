"""Read-only descriptive analysis; no optimizer or model selection."""
from collections import defaultdict
from pathlib import Path
import json
from data import TEMPLATES
from experiment import aggregate
from trainlab.runner import write_json,digest

root=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (root/'results/predictions.jsonl').read_text().splitlines()]
report=json.loads((root/'results/summary.json').read_text())
for s in report['results']:
    selected=[r for r in rows if (r['arm'],r['seed'],r['condition'])==(s['arm'],s['seed'],s['condition'])]
    if aggregate(selected)!=s['overall']:raise ValueError('aggregate mismatch')
    for t in TEMPLATES:
        part=[r for r in selected if r['template']==t]
        if aggregate(part)!=s['templates'][t]:raise ValueError('template mismatch')
        for op in ('copy','link'):
            if aggregate([r for r in part if r['focal_op']==op])!=s['template_focal'][t][op]:raise ValueError('focal mismatch')

means={}
for arm in ('csm','pointer_slot'):
    means[arm]={}
    for condition in ('normal','erase_values','identity_links','supplied_control'):
        items=[s for s in report['results'] if s['arm']==arm and s['condition']==condition]
        means[arm][condition]={'per_seed':{str(s['seed']):s['overall'] for s in items},
            'mean_accuracy':sum(s['overall']['accuracy'] for s in items)/3,
            'mean_complete_map':sum(s['overall']['complete_map_accuracy'] for s in items)/3,
            'per_template':{t:{'mean_accuracy':sum(s['templates'][t]['accuracy'] for s in items)/3,
                'per_seed_accuracy':{str(s['seed']):s['templates'][t]['accuracy'] for s in items},
                'every_seed_at_least_95pct':all(s['templates'][t]['accuracy']>=.95 for s in items)} for t in TEMPLATES},
            'query_preserved_fraction':sum(s['query_preservation']['correct'] for s in items)/sum(s['query_preservation']['total'] for s in items)}
    critical={}
    for template,role in [('overwrite','alias'),('delete_gap','alias'),('recreate_snapshot','snapshot'),('detach','alias'),('delete_alias','alias')]:
        critical[template]={}
        for seed in (101,202,303):
            part=[r for r in rows if r['arm']==arm and r['condition']=='normal' and r['seed']==seed and r['template']==template and r['role']==role]
            critical[template][str(seed)]={'correct':sum(r['correct'] for r in part),'total':len(part)}
    means[arm]['critical_role_outcomes']=critical
write_json(root/'results/analysis.json',{'means':means,'prediction_count':len(rows),
    'raw_prediction_sha256':digest(root/'results/predictions.jsonl'),'uncertainty':'DESCRIPTIVE_SEEDS_ONLY; FIVE_KNOWN_TEMPLATES'})
print(json.dumps(means,indent=2))
