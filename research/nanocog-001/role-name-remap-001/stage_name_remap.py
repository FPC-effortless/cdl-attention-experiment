"""Freeze two native label-free actor recipes; does not import/run a model."""
from hashlib import sha256
from pathlib import Path
import json,shutil
from trainlab.runner import load_config,verify_run

R=Path(__file__).resolve().parent
P=R.parent/'cog-role-repair-004'
OL=R.parent/'cog-role-002'
A='0775d4358384f12486d616988aefa8a764888ece48e8942f96c617754ff07514'
J='3e1eecfc644f11bad7c722cce6280a39f2650fe93d750a1a9fb52a0bc7d42947'
ML='a448f743e30f846f72be1c1980a53256d407dd1a942c3d22b22c614b73f64a4c'
W='620abcff0287efb6b8087d5057de3cbbb04a08a7f864960b223e98bd170c3137'
PROTOCOL='422e83beec44d8aaab1f2fd1293fbcb6f63ffb3ec4c89082b22c185d499a3f67'
B='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
AD='abfd950297ca1b90846621f770601b7e38081f5c62eea49e4e1c814477674932'
L='317019d1cd792b6c62625deef81566a8297d0b90b0998160749797322ae98d37'
PS='0971d030ad7b74f7456e5eabdfa39b87dcd3641d90b6c2dfbf9e31e51366f1da'
FS='2b231b0e918e561f04adf7e0375c37bc3caa72090d9fd954a5ad4702600ecfed'
def req(ok,m):
    if not ok:raise ValueError(m)
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
    b=p.read_bytes();req(pin is None or sha256(b).hexdigest()==pin,'pin '+str(p))
    return json.loads(b)
def put(p,v):
    with p.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False,indent=2);f.write('\n')
if __name__=='__main__':
    m=read(R/'panel/manifest.json',ML)
    req((m['actor_sha256'],m['judge_sha256'],m['native_backbone_sha256'],m['tap1471_adapter_sha256'])==(A,J,B,AD)
        and sha(R/'panel/actor.json')==A and sha(R/'panel/judge.json')==J
        and sha(R/'predict_name_remap.py')==W and sha(R/'NAME_REMAP_PROTOCOL.md')==PROTOCOL,'reviewed input bytes')
    panel=read(R/'panel/actor.json',A)
    req(set(panel)=={'schema','prompts'} and panel['schema']=='cog-role-name-remap.actor.v1'
        and len(set(panel['prompts']))==len(panel['prompts'])==768,'label-free projection')
    prev=P/'repair_actor_admissions';old=read(prev/'ledger.json',L)
    prevseal=read(P/'REPAIR_PREDICTION_SEAL.json',PS)
    req(prevseal['ledger_sha256']==L,'previous actor ledger identity')
    fit=read(P/'FIT_SEAL.json',FS)['runs']['tap_compare-1471']
    f=P/'fit_runs'/fit['id']
    fr=read(f/'run.json',fit['receipt_sha256'])
    req(verify_run(f)['valid'] and fr['status']=='succeeded' and fr['research_adequacy']=='EXECUTED_CRITERIA_MET'
        and fr['artifacts']['adapter-1280.pt']==fit['artifacts']['adapter-1280.pt']==sha(f/'output/adapter-1280.pt')==AD,'frozen adapter provenance')
    base=read(OL/'ROLE002_FIT_INDEPENDENT_AUDIT.json','92d9bbf3b1b4643236be235fa7c9292dcceb29de7aeb65eca5a00ce3e2cf880c')['runs']['1371']
    b=OL/'fits'/base['run_id'];br=read(b/'run.json',base['run_sha256'])
    req(verify_run(b)['valid'] and br['status']=='succeeded' and br['artifacts']['checkpoint-1280.pt']
        ==sha(b/'output/checkpoint-1280.pt')==B,'frozen base provenance')
    h=R/'remap_admissions';h.mkdir()
    rows={}
    for tag,prev_tag,arm,seed,ep,ap in [
        ('remap-native-1371','native-1371-0','native',1371,0,'none'),
        ('remap-tap-1471','tap_compare-1471-1280','tap_compare',1471,1280,AD)]:
        c=old[prev_tag]; prevd=prev/prev_tag; ss=prevseal['results'][prev_tag]
        rd=P/'repair_actor_runs'/ss['id'];rr=read(rd/'run.json',ss['receipt_sha256'])
        req(verify_run(rd)['valid'] and rr['status']=='succeeded' and rr['inputs']==c['inputs']
          and c['arm']==arm and c['seed']==seed and c['endpoint']==ep and c['backbone_sha256']==B
          and c['adapter_sha256']==ap,'prior replay provenance')
        keep={n:prevd/n for n in c['inputs'] if n not in ('actor.json','predict_adapter.py')}
        req(all(sha(p)==c['inputs'][n] for n,p in keep.items())
           and sha(keep['checkpoint.pt'])==B and (ap=='none')==('adapter.pt' not in keep)
           and (ap=='none' or sha(keep['adapter.pt'])==AD),'shared model sources')
        keep.update({'actor.json':R/'panel/actor.json','predict_name_remap.py':R/'predict_name_remap.py',
                     'NAME_REMAP_PROTOCOL.md':R/'NAME_REMAP_PROTOCOL.md'})
        req(not any(n in {'judge.json','train.json','labels.json','predictions.json','results.json'} for n in keep),'actor custody')
        d=h/tag;d.mkdir()
        for n,p in keep.items():
            o=d/n;o.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,o)
        cfg=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',name=tag,
            description='Frozen Role identity-remapping label-free actor only; no judge or updates.',
            source_kind='CONSTRUCTED',split_identity='ROLE-NAME-REMAP-001-fresh-combination-actor',
            command=['python','predict_name_remap.py',arm,str(seed),str(ep),ap],seed=seed,timeout_seconds=45,
            inputs=sorted(keep),artifacts=['predictions.json'],budget={'inference_batches':96},
            metrics={'prediction_rows':'max','model_forwards':'min','optimizer_updates':'min'})
        put(d/'recipe.json',cfg)
        req(load_config(d/'recipe.json')==cfg,'native recipe validation changed')
        rows[tag]=dict(seed=seed,arm=arm,endpoint=ep,base_sha256=B,adapter_sha256=ap,actor_sha256=A,
          recipe_sha256=sha(d/'recipe.json'),prior_fit_receipts={'base':base['run_sha256'],'adapter':fit['receipt_sha256'] if ap!='none' else None},
          prior_actor_receipt_sha256=ss['receipt_sha256'],inputs={n:sha(d/n) for n in keep})
    put(h/'ledger.json',rows)
    print(sha(h/'ledger.json'))
