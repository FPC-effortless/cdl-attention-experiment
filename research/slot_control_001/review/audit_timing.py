"""Read-only independent timing-record audit. Does not rerun timing."""
from collections import defaultdict
import hashlib,json,math
from pathlib import Path
from statistics import median,mean
ROOT=Path(__file__).resolve().parents[1]
STRESS=ROOT.parent/'changing_state_001'
def check(c,m):
    if not c:raise RuntimeError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    timing=json.loads((ROOT/'results/timing.json').read_text());frozen=json.loads((ROOT/'freeze.json').read_text())
    check(timing['freeze_sha256']==sha(ROOT/'freeze.json'),'timing freeze')
    check(sha(ROOT/'timing.py')==frozen['files']['timing.py'],'timing source')
    check(timing['threads']==1 and timing['batch']==1 and timing['repetitions']==3 and timing['warmup']==1,'recipe')
    worlds=json.loads((STRESS/'worlds.json').read_text())
    lookup={(w['block'],w['regime'],w['updates']):w for w in worlds if w['family'].endswith('-s0')}
    rows=timing['measurements'];check(len(rows)==864,'row count');groups=defaultdict(dict)
    expected={(s,b,g,d,r) for s in [101,202,303] for b in range(8) for g in ['all_changed','one_changed'] for d in [1,16,256] for r in range(3)}
    for index,row in enumerate(rows):
        identity=tuple(row[k] for k in ('seed','block','regime','updates','rep'))
        check(identity in expected,'unexpected identity')
        check(row['model'] in ['csm','slot'] and row['model'] not in groups[identity],'duplicate/model')
        check(type(row['elapsed_ns']) is int and row['elapsed_ns']>0 and math.isfinite(row['elapsed_ns']),'duration')
        check(row['query']==row['block']%4,'query selection')
        correct=11+dict(lookup[(row['block'],row['regime'],row['updates'])]['writes'])[row['query']]
        check(row['answer']==correct and row['prediction']==correct,'correctness')
        order=['csm','slot'] if (row['block']+row['rep']+row['seed'])%2 else ['slot','csm']
        check(row['order']==order,'declared order')
        check(row['model']==order[index%2],'actual serialized order')
        if index%2:check(identity==tuple(rows[index-1][k] for k in ('seed','block','regime','updates','rep')),'nonadjacent pair')
        groups[identity][row['model']]=row['elapsed_ns']
    check(set(groups)==expected and all(set(v)=={'csm','slot'} for v in groups.values()),'missing pairs')
    pairs=[{'seed':k[0],'block':k[1],'regime':k[2],'updates':k[3],'rep':k[4],
            'csm_ns':v['csm'],'slot_ns':v['slot'],'csm_over_slot':v['csm']/v['slot']} for k,v in groups.items()]
    pooled=[];cells=[]
    for depth in [1,16,256]:
        selected=[p for p in pairs if p['updates']==depth]
        cellratios=[]
        for seed in [101,202,303]:
            for regime in ['all_changed','one_changed']:
                cell=[p for p in selected if p['seed']==seed and p['regime']==regime]
                ratio=median(p['csm_over_slot'] for p in cell);cellratios.append(ratio)
                cells.append({'seed':seed,'regime':regime,'updates':depth,'paired_samples':len(cell),
                    'csm_median_ms':median(p['csm_ns'] for p in cell)/1e6,'slot_median_ms':median(p['slot_ns'] for p in cell)/1e6,
                    'median_paired_csm_over_slot':ratio})
        pooled.append({'updates':depth,'paired_samples':len(selected),
            'csm_median_ms':median(p['csm_ns'] for p in selected)/1e6,
            'slot_median_ms':median(p['slot_ns'] for p in selected)/1e6,
            'median_paired_csm_over_slot':median(p['csm_over_slot'] for p in selected),
            'ratio_of_pooled_medians':median(p['csm_ns'] for p in selected)/median(p['slot_ns'] for p in selected),
            'seed_regime_median_ratio_min':min(cellratios),'seed_regime_median_ratio_max':max(cellratios),
            'single_pair_min':min(p['csm_over_slot'] for p in selected),'single_pair_max':max(p['csm_over_slot'] for p in selected)})
    outliers=[p for p in pairs if p['csm_over_slot']<1]
    original=json.loads((STRESS/'results/summary.json').read_text())
    means={arm:100*mean(r['accuracy'] for r in original if r['arm']==arm and r.get('status')!='NOT_SCORED') for arm in ['raw','transformer']}
    return {'status':'PASS','rows':len(rows),'pairs':len(pairs),'pooled_by_depth':pooled,'seed_regime_cells':cells,
            'pairs_where_slot_slower':outliers,'raw_reference_mean_accuracy_percent':means,
            'timing_sha256':sha(ROOT/'results/timing.json'),'timing_source_sha256':sha(ROOT/'timing.py'),
            'freeze_sha256':sha(ROOT/'freeze.json'),'report_sha256_at_review':sha(ROOT.parent/'CHANGING_STATE_001_REPORT.md'),
            'definitions':{'paired_ratio':'For each seed/block/regime/depth/rep, CSM elapsed_ns divided by slot elapsed_ns; then median.',
                           'pooled_median':'Median over144 durations per model/depth, pooling3seeds x2regimes x8blocks x3repetitions.',
                           'seed_regime_range':'Min/max over six cells per depth; each cell median of24paired ratios.'},
            'no_timing_rerun':True}
if __name__=='__main__':
    result=main();(ROOT/'review/timing_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pooled':result['pooled_by_depth'],'slot_slower_pairs':result['pairs_where_slot_slower'],'means':result['raw_reference_mean_accuracy_percent']},indent=2))
