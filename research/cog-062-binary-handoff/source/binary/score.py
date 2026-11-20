"""Frozen independent scoring rule, no training or checkpoint load."""
from pathlib import Path
from collections import defaultdict
import json,os,sys
import numpy as np

ARMS=('base','canonical','ordered','order_removed')
HOLD='<hold>'
EMPTY='<empty>'
key=lambda r:(r['group'],r['dialogue'],r['turn'])

def reduce(records,rng_seed=62062):
    n=len(records)
    by=defaultdict(lambda:np.zeros(1+len(ARMS),dtype=np.int64))
    for group,y,p in records:
        row=by[group];row[0]+=1
        for i,tag in enumerate(ARMS,1): row[i]+=int(p[tag]==y)
    cnt=sum(by.values(),start=np.zeros(1+len(ARMS),dtype=np.int64))
    supported=(n>=100 and len(by)>=50)
    out={'rows':n,'groups':len(by),'support':('MINIMUM_MET' if supported else 'INSUFFICIENT_SUPPORT'),'accuracy':{tag:{'correct':int(cnt[i]),'total':n} for i,tag in enumerate(ARMS,1)}}
    if supported:
        g=np.stack([by[k] for k in sorted(by)]);rng=np.random.default_rng(rng_seed);draws=[]
        for _ in range(25):
            draws.extend(g[rng.integers(len(g),size=(200,len(g)))].sum(axis=1))
        b=np.stack(draws);out['paired']={}
        for first,second in (('ordered','canonical'),('ordered','base'),('ordered','order_removed')):
            i,j=(1+ARMS.index(k) for k in (first,second));out['paired'][first+'_minus_'+second]={'accuracy_difference':float((cnt[i]-cnt[j])/n),'descriptive_dialogue_bootstrap_95':np.quantile((b[:,i]-b[:,j])/b[:,0],[.025,.975]).tolist()}
    return out

def main(pred_path,gold_path,overlap_path):
    rows=json.loads(Path(pred_path).read_bytes());gold=json.loads(Path(gold_path).read_bytes()); overlap=json.loads(Path(overlap_path).read_bytes())
    if len(rows)!=len(gold) or len({key(r) for r in rows})!=len(rows) or [key(r) for r in rows]!=[key(r) for r in gold] or [key(r) for r in rows]!=[key(r) for r in overlap]:
        raise ValueError('prediction/gold row identity mismatch')
    cohorts=defaultdict(list);miss=top2=skipped=0;unchanged=defaultdict(lambda:defaultdict(int));switches={tag:dict(total=0,rescued_rank2=0,spoiled_rank1=0,unchanged_false_update=0,unchanged_other_switch=0,both_miss_switch=0) for tag in ARMS[1:]}
    for r,g,flags in zip(rows,gold,overlap):
        if any(type(flags.get(k)) is not bool for k in ('group_contains_tuple_overlap','exact_tuple_overlap')): raise ValueError('invalid overlap tags')
        if len(r['slots'])!=2 or len(g['delta'])!=2:raise ValueError('wrong slot count')
        for j,(p,y,old) in enumerate(zip(r['slots'],g['delta'],g['prior'])):
            if set(p)!=set(ARMS[1:])|{'top2'} or len(p['top2'])!=2 or len(set(p['top2']))!=2 or any(p[a] not in p['top2'] for a in ARMS[1:]):raise ValueError('invalid shortlist outputs')
            if y is None:skipped+=1;continue
            p['base']=p['top2'][0];record=(r['group'],y,p)
            slot=('area','price')[j]
            for tag,v in switches.items():
                if p[tag]==p['base']:continue
                v['total']+=1
                if y==HOLD: cat='unchanged_false_update' if p[tag]!=HOLD else 'unchanged_other_switch'
                elif y==p[tag]: cat='rescued_rank2'
                elif y==p['base']: cat='spoiled_rank1'
                else: cat='both_miss_switch'
                v[cat]+=1
            if y==HOLD:
                for tag in ARMS:unchanged[slot][tag]+=int(p[tag]!=HOLD)
                unchanged[slot]['eligible']+=1
                cohorts['unchanged'].append(record)
                continue
            rank=p['top2'].index(y)+1 if y in p['top2'] else 0
            top2+=int(rank>0);miss+=int(rank==0)
            kind='removal' if y==EMPTY else 'first_set' if old==EMPTY else 'overwrite'
            cohorts['candidate:hit' if rank else 'candidate:miss'].append(record)
            if rank:cohorts['candidate:hit:'+slot].append(record)
            for tag in ('changed','slot:'+slot,'kind:'+kind,'slot:'+slot+':kind:'+kind,'rank:'+str(rank),'history:'+str(min(2,r['turn']))):
                cohorts[tag].append(record)
            cohorts['overlap:group' if flags['group_contains_tuple_overlap'] else 'overlap:clean_group'].append(record)
            cohorts['overlap:tuple' if flags['exact_tuple_overlap'] else 'overlap:clean_tuple'].append(record)
    tags={'candidate:hit','candidate:miss','candidate:hit:area','candidate:hit:price','changed','slot:area','slot:price','unchanged'}|{k for q in ('first_set','overwrite','removal') for k in ('kind:'+q,'slot:area:kind:'+q,'slot:price:kind:'+q)}|{'rank:0','rank:1','rank:2','history:0','history:2','overlap:group','overlap:clean_group','overlap:tuple','overlap:clean_tuple'}
    counts={k:reduce(cohorts[k]) for k in sorted(tags)}
    c=counts['changed'];adequate=c['rows']>=400 and c['groups']>=250 and all(counts['slot:'+x]['rows']>=120 for x in ('area','price'))
    p=c.get('paired',{}).get('ordered_minus_canonical')
    positive=bool(adequate and p and p['accuracy_difference']>=.03 and p['descriptive_dialogue_bootstrap_95'][0]>0)
    fixes={}
    for tag in ARMS[1:]:
        fixes[tag]={'base_wrong_now_right':sum(p['base']!=y and p[tag]==y for _,y,p in cohorts['changed']),'base_right_now_wrong':sum(p['base']==y and p[tag]!=y for _,y,p in cohorts['changed'])}
    out={'scope':'HISTORICALLY_EXPOSED_TASK_PROSPECTIVE_WITHIN_CORPUS_DIAGNOSTIC_ONLY','qualification':'NOT_QUALIFIED','rows':len(rows),'ineligible_slot_rows':skipped,'changed_top2_oracle':top2,'changed_shortlist_miss':miss,'minimum_primary_support_met':adequate,'directional_gate_met':positive,'cohorts':counts,'flip_tradeoff':fixes,'total_rank2_switch_accounting':switches,'unchanged_false_update_proposals':unchanged,'not_assessed':['verified state writes','full-request inference economics','external transfer or uncontaminated benchmark','explicit spoken cancellation']}
    dest=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    (dest/'scores.json').write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n')
    (dest/'metrics.json').write_text(json.dumps({'scored_turns':len(rows)}))
if __name__=='__main__': main(*sys.argv[1:])
