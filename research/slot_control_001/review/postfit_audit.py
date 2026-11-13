"""Independent raw-record metric reconstruction and sampled frozen replay."""
from collections import defaultdict
import hashlib,importlib.util,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STRESS=ROOT.parent/'changing_state_001'
SOURCE=Path('/workspace/scratch/a74a27d116f3/recovered')
ADDRESS=SOURCE/'experiments/csm_address_002'

def check(test,message):
    if not test:raise RuntimeError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def key(r):return (r['arm'],r['seed'],r['regime'],r['updates'],r['family'],r['query'])
def cell(r):return (r['arm'],r['seed'],r['regime'],r['updates'])
def metrics(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['family']].append(r)
    check(all(len(v)==4 and len({r['query'] for r in v})==4 for v in groups.values()),'incomplete world')
    changed=[r for r in rows if r['changed']];other=[r for r in rows if not r['changed']]
    def acc(xs):return sum(r['prediction']==r['answer'] for r in xs)/len(xs) if xs else None
    stale=sum(r['prediction']==r['prior'] for r in changed)
    return {'queries':len(rows),'accuracy':acc(rows),'worlds':len(groups),
        'complete_world_accuracy':sum(all(r['prediction']==r['answer'] for r in group) for group in groups.values())/len(groups),
        'changed_queries':len(changed),'changed_accuracy':acc(changed),'stale_errors':stale,
        'stale_rate':stale/len(changed) if changed else None,'untouched_queries':len(other),'untouched_accuracy':acc(other)}

def main():
    start=time.perf_counter(); hashes={}
    for base in (ROOT,STRESS):
        frozen=load(base/'freeze.json')
        for name,h in frozen['files'].items():check(sha(base/name)==h,'frozen source/data mismatch: '+name)
        hashes[str(base.name)+'/freeze.json']=sha(base/'freeze.json')
    for name,h in load(STRESS/'freeze.json')['dependencies'].items():
        check(sha(SOURCE/name)==h,'original dependency mismatch: '+name)
    closure=load(ROOT/'training_closed.json')
    check(closure['all_three_closed_before_stress_scoring'] is True,'closure order flag')
    check(sorted(r['seed'] for r in closure['runs'])==[101,202,303],'closure seeds')
    for receipt in closure['runs']:
        d=ROOT/'runs'/str(receipt['seed']);check(receipt==load(d/'complete.json'),'closure/run mismatch')
        check(receipt['freeze_sha256']==sha(ROOT/'freeze.json'),'fit source mismatch')
        check(receipt['updates']==256 and receipt['answer_exposures']==8192,'budget')
        check(sha(d/'checkpoint.pt')==receipt['checkpoint_sha256'],'fit checkpoint hash')
        check(sha(d/'updates.jsonl')==receipt['training_log_sha256'],'fit log hash')
        rows=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
        check([r['step'] for r in rows]==list(range(1,257)),'step order')
        check(all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in rows),'nonfinite training')
        orig=load(ADDRESS/'runs'/f"supervised_{receipt['seed']}"/'complete.json')
        check(receipt['schedule_sha256']==orig['schedule_sha256'],'schedule mismatch')
    worlds=load(STRESS/'worlds.json');truth={}
    for w in worlds:
        current={};previous={}
        for k,v in w['writes']:
            if k in current:previous[k]=current[k]
            current[k]=v
        for q in range(4):
            truth[(w['regime'],w['updates'],w['family'],q)]={'answer':11+current[q],
                'prior':11+previous[q] if q in previous else None,'changed':q in previous,'block':w['block']}
    # Historical vocabulary has v0 at index11; verify independently from its file.
    sys.path.insert(0,str(ADDRESS));import data as original_data
    check(original_data.VOCABULARY['v0']==11,'vocabulary offset')
    all_records={};all_summaries=[];row_counts={}
    for base in (ROOT,STRESS):
        out=base/'results';receipt=load(out/'receipt.json')
        check(receipt['predictions_sha256']==sha(out/'predictions.jsonl'),'prediction hash')
        check(receipt['freeze_sha256']==sha(base/'freeze.json'),'result source mismatch')
        records=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
        row_counts[base.name]=len(records);grouped=defaultdict(list)
        for r in records:
            check(key(r) not in all_records,'duplicate record');all_records[key(r)]=r
            expected=truth[(r['regime'],r['updates'],r['family'],r['query'])]
            check(all(r[k]==v for k,v in expected.items()),'semantic metadata mismatch')
            check(r['correct']==(r['prediction']==r['answer']),'correctness field')
            grouped[cell(r)].append(r)
        summary=load(out/'summary.json');seen=set()
        for s in summary:
            k=cell(s);check(k not in seen,'duplicate metric cell');seen.add(k)
            if s.get('status')=='NOT_SCORED':
                check(s['arm']=='transformer' and s['updates'] in [64,256] and k not in grouped,'invalid unscored cell')
                continue
            check(k in grouped,'missing metric cell')
            observed=metrics(grouped[k]);check(all(s[k]==v for k,v in observed.items()),'metric mismatch')
            check(observed['queries']==256 and observed['worlds']==64,'panel size')
            all_summaries.append({'arm':s['arm'],'seed':s['seed'],'regime':s['regime'],'updates':s['updates'],**observed})
        check(set(grouped)<=seen,'unreported records')
        hashes[f'{base.name}/predictions.jsonl']=sha(out/'predictions.jsonl')
    check(row_counts=={'slot_control_001':7680,'changing_state_001':27648},'prediction completeness')
    check(len(all_summaries)==138,'metric row count')
    # Also independently reconstruct the three deterministic policies.
    controls=load(STRESS/'results/controls.json')
    for s in controls:
        rows=[]
        for w in worlds:
            if w['regime']!=s['regime'] or w['updates']!=s['updates']:continue
            for q in range(4):
                r={'family':w['family'],'query':q,**truth[(w['regime'],w['updates'],w['family'],q)]}
                if s['policy']=='current_value_oracle':pred=r['answer']
                elif s['policy']=='initial_value':pred=11+dict(w['writes'][:4])[q]
                elif s['policy']=='last_global_value':pred=11+w['writes'][-1][1]
                else:raise RuntimeError('unknown policy')
                r['prediction']=pred;rows.append(r)
        check(all(s[k]==v for k,v in metrics(rows).items()),'control metric mismatch')
    # Frozen native replay: all queries for eight s0 blocks at depths1/256,
    # both regimes, every seed and CSM arm; Transformer has only depth1 here.
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import model_adapter as adapter
    spec=importlib.util.spec_from_file_location('slot_postfit_review_model',ROOT/'model.py')
    slotmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(slotmod)
    replay=[]
    with torch.inference_mode():
        for arm in ('slot_store','supervised','raw','supplied_controller','transformer'):
            for seed in (101,202,303):
                if arm=='slot_store':
                    path=ROOT/'runs'/str(seed)/'checkpoint.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
                    check(saved['seed']==seed and saved['updates']==256,'checkpoint metadata')
                    check(all(float(state['step'])==256 for state in saved['optimizer']['state'].values()),'optimizer steps')
                    model=slotmod.SlotStore()
                else:
                    path=ADDRESS/'runs'/f'{arm}_{seed}'/'checkpoint.pt'
                    check(sha(path)==load(STRESS/'freeze.json')['checkpoints'][f'{arm}_{seed}'],'original checkpoint hash')
                    saved=torch.load(path,map_location='cpu',weights_only=False);model=adapter.build_model(arm,seed,19)
                model.load_state_dict(saved['model']);model.eval()
                before={k:v.clone() for k,v in model.state_dict().items()}
                count=0
                for regime in ('all_changed','one_changed'):
                    for depth in ([1] if arm=='transformer' else [1,256]):
                        selected=[w for w in worlds if w['regime']==regime and w['updates']==depth and w['family'].endswith('-s0')]
                        check(len(selected)==8,'replay block count')
                        examples=[];ids=[]
                        for w in selected:
                            for q in range(4):
                                examples.append(original_data.encode({'writes':w['writes'],'world_id':w['family'],'family_id':w['family']},q))
                                ids.append((arm,seed,regime,depth,w['family'],q))
                        batch=adapter.tensors(examples)
                        if arm=='slot_store':logits,_=model.sequence(batch['ids'])
                        else:logits,_=adapter.run_model(model,arm,batch)
                        check(torch.isfinite(logits).all(),'nonfinite replay')
                        preds=logits.argmax(-1).tolist()
                        for ident,pred in zip(ids,preds):check(pred==all_records[ident]['prediction'],'native replay mismatch')
                        count+=len(preds)
                check(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()),'replay altered weights')
                replay.append({'arm':arm,'seed':seed,'predictions_replayed':count,'checkpoint_sha256':sha(path)})
    report={'status':'PASS','raw_prediction_counts':row_counts,'metric_cells_recomputed':len(all_summaries),
        'control_cells_recomputed':len(controls),'replayed_predictions':sum(r['predictions_replayed'] for r in replay),
        'replay':replay,'source_and_prediction_hashes':hashes,'training_updates':768,'answer_exposures':24576,
        'elapsed_s':time.perf_counter()-start,'torch':torch.__version__,
        'limits':['Eight independent base blocks; shifts/depths/regimes/queries correlated.',
                  'Historical Transformer answer-only supervision differs.',
                  'Aggregate controller counters were not independently replayed in full.',
                  'Native replay is sampled, not all predictions; all saved metrics are independently recomputed.',
                  'No fits, timing, scientific promotion, external replication or broad efficiency claim.']}
    output=ROOT/'review/postfit_checks.json'
    with output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True)
    print(json.dumps({'status':'PASS','metric_cells':len(all_summaries),'raw_predictions':sum(row_counts.values()),'replayed':report['replayed_predictions'],'elapsed_s':report['elapsed_s']},indent=2),flush=True)

if __name__=='__main__':main()
