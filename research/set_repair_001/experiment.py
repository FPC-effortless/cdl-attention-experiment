"""Twelve fixed-budget continuations; evaluate only after the complete fit closure."""
from pathlib import Path
import sys,json,hashlib,os,random,time,shutil
from datetime import datetime
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
from model_source import PARENT,PINS,build,tensors,encode,history_map,VOCAB,TOKENS,HEADS,CUTOFF
from paired_data import generate,validate,example,KEYS,TEMPLATES
from probe_grid import generate as probe_generate
from learn import forward_with_root,loss_parts
import torch,numpy as np
from forge_data.core import canonical,release
from forge_data.dataset import Dataset
from trainlab.runner import digest,read_json,write_json,run_experiment,verify_run
from research_lab.ecosystem import _runtime
from research_lab.core import Controller
from research_lab.adapters import import_forge,import_trainlab
from research_lab.evidence import put_node,link,retrieve
from investigation_world.benchmark.lifecycle_observability import replay
from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceSubjectRef,EvidenceProducerRef,EvidenceArtifactRef
ARMS=('csm','pointer_slot');SEEDS=(101,202,303);STAGES=('control','invariance')

def require(x,message):
    if not x:raise ValueError(message)

def runtime():return {**_runtime(),'torch':torch.__version__,'torch_file':str(Path(torch.__file__).resolve())}

def parent_pins():
    require(digest(PARENT/'freeze.json')=='449e0aa06076e99547ad78f70a8a9206c6887cd2346fde1a25d7a085bf309d33','parentfreeze')
    require(digest(PARENT/'training_closed.json')=='a181cb5c3fca3d34a761c98cf2ca03f282618fccc9cef0fca73b634dc49deaac','parentclosure')
    f=read_json(PARENT/'freeze.json');require(f['runtime']==runtime(),'parentruntime')
    pins={'freeze.json':digest(PARENT/'freeze.json'),'training_closed.json':digest(PARENT/'training_closed.json')}
    for name,h in PINS.items():require(digest(PARENT/(name+'.py'))==h,'parentsource');pins[name+'.py']=h
    for r in read_json(PARENT/'training_closed.json')['runs']:
        p=f"runs/{r['name']}/{r['run_id']}";require(verify_run(PARENT/p)['valid'],'parentreceipt')
        require(digest(PARENT/p/'output/checkpoint.pt')==r['checkpoint_sha256'],'parentweight')
        pins[p+'/output/checkpoint.pt']=r['checkpoint_sha256'];pins[p+'/run.json']=digest(PARENT/p/'run.json')
    return pins

def check():
    f=read_json(ROOT/'freeze.json');require(f['runtime']==runtime() and f['parent_pins']==parent_pins(),'runtime/parent drift')
    for n,h in f['files'].items():require(digest(ROOT/n)==h,'source drift '+n)
    return f

def ds():
    f=check();return Dataset(ROOT/'dataset',manifest_sha256=f['manifest_sha256'])

def schedule(seed,n):
    g=torch.Generator().manual_seed(seed+700001)
    return torch.randint(n,(256,16),generator=g),torch.randint(4,(256,16),generator=g)

def make_bank(pairs):
    rows=[example(w,q) for p in pairs for q in KEYS for w in p['worlds']]
    raw=tensors(rows)
    return {k:v.reshape((len(pairs),4,2)+v.shape[1:]) for k,v in raw.items()}

def batch_at(bank,pairs,index,queries):
    batch={k:v[index,queries].flatten(0,1) for k,v in bank.items()}
    roots=torch.tensor([int(pairs[i]['root'][1:]) for i in index.tolist()]).repeat_interleave(2)
    require(bool((batch['ids'][:,15]==VOCAB['set']).all()),'wrong SET token')
    require(bool((batch['ids'][:,16]==roots+VOCAB['r0']).all()),'wrong root')
    return batch,roots

def prepare():
    require(not (ROOT/'freeze.json').exists(),'alreadyfrozen');pins=parent_pins();rows=generate();audit=validate(rows,replay)
    (ROOT/'initial').mkdir()
    for r in read_json(PARENT/'training_closed.json')['runs']:shutil.copyfile(PARENT/'runs'/r['name']/r['run_id']/'output/checkpoint.pt',ROOT/'initial'/(r['name']+'.pt'))
    (ROOT/'source.jsonl').write_bytes(b''.join(canonical(r) for r in rows))
    write_json(ROOT/'source_spec.json',{'dataset_id':'set-repair-001','source_id':'generated/set-repair-001','source_version':digest(ROOT/'paired_data.py'),
        'adapter':'operations-jsonl','license_id':'GENERATED_RESEARCH_DEV','provenance_uri':'urn:woyengi:set-repair-001','ai_use_allowed':True,
        'redistribution_allowed':True,'privacy_reviewed':True,'split_salt':'set-repair-001-fixed-v1'})
    release(ROOT/'source.jsonl',ROOT/'source_spec.json',ROOT/'dataset');manifest=digest(ROOT/'dataset/manifest.json')
    data=Dataset(ROOT/'dataset',manifest_sha256=manifest)
    vp=[r['features']['pair'] for r in data.examples('validation',cutoff=CUTOFF,feature_fields=['pair'])]
    require({p['new'] for p in vp}=={f'v{i}' for i in range(8)},'validation value coverage')
    require({(p['template'],p['focal_op']) for p in vp}=={(t,o) for t in TEMPLATES for o in ('copy','link')},'validation stratum coverage')
    write_json(ROOT/'preparation.json',{'semantic_audit':audit,'forge':data.summary(),'parent_pins':pins,'fits_planned':12})
    files=[p for p in ROOT.iterdir() if p.suffix in ('.py','.md','.json','.jsonl')]+list((ROOT/'dataset').iterdir())+list((ROOT/'initial').iterdir())
    write_json(ROOT/'freeze.json',{'files':{p.relative_to(ROOT).as_posix():digest(p) for p in files},'parent_pins':pins,'runtime':runtime(),'manifest_sha256':manifest,'updates':256,'pairs_per_batch':16})
    recipes={}
    for seed in SEEDS:
        for arm in ARMS:
            for stage in STAGES:
                name=f'{arm}-{seed}-{stage}'
                recipe={'schema':'trainlab.experiment.v2','name':name,'description':'Matched continuation objective test',
                    'command':[sys.executable,'experiment.py','train',arm,str(seed),stage],'seed':seed,'timeout_seconds':180,
                    'source_kind':'CONSTRUCTED','split_identity':manifest,'inputs':[p.relative_to(ROOT).as_posix() for p in files]+['freeze.json'],
                    'artifacts':['checkpoint.pt','updates.jsonl','complete.json'],'metrics':{'updates':'max','final_loss':'min'},
                    'budget':{'updates':256,'pairs_per_batch':16,'answer_exposures':8192,'cpu_threads':1},
                    'workload':{'kind':'train','training_method':'supervised','execution':'single_process'},'checks':{}}
                write_json(ROOT/(name+'.json'),recipe);recipes[name+'.json']=digest(ROOT/(name+'.json'))
    write_json(ROOT/'dispatch.json',{'freeze':digest(ROOT/'freeze.json'),'recipes':recipes})
    return {'dispatch_sha256':digest(ROOT/'dispatch.json'),'dataset':data.summary()}

def train(arm,seed,stage):
    check();require(arm in ARMS and seed in SEEDS and stage in STAGES,'unknownfit');require(not (ROOT/'training_closed.json').exists(),'closed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(seed);torch.manual_seed(seed)
    pairs=[r['features']['pair'] for r in ds().examples('train',cutoff=CUTOFF,feature_fields=['pair'])]
    bank=make_bank(pairs);ii,qq=schedule(seed,len(pairs));schedhash=hashlib.sha256(ii.numpy().tobytes()+qq.numpy().tobytes()).hexdigest()
    initial=ROOT/'initial'/f'{arm}-{seed}.pt';model=build(arm);model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['model']);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);started=time.perf_counter()
    with (out/'updates.jsonl').open('x') as log:
        for step,(index,queries) in enumerate(zip(ii,qq),1):
            batch,roots=batch_at(bank,pairs,index,queries);optimizer.zero_grad(set_to_none=True)
            logits,cap,states=forward_with_root(model,batch,roots)
            loss,parts=loss_parts(logits,cap,states,batch,1.0 if stage=='invariance' else 0.0)
            require(bool(torch.isfinite(loss)),'nonfinite loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            require(bool(torch.isfinite(norm)),'nonfinite gradient');optimizer.step()
            item={'step':step,'loss':float(loss.detach()),'parts':parts,'gradient_norm':float(norm),'batch_accuracy':float((logits.argmax(-1)==batch['answer']).float().mean())}
            log.write(json.dumps(item)+'\n');log.flush()
            if step%64==0:print(arm,seed,stage,step,item['batch_accuracy'],flush=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'arm':arm,'seed':seed,'stage':stage,'updates':256},out/'checkpoint.pt')
    complete={'arm':arm,'seed':seed,'stage':stage,'updates':256,'answer_exposures':8192,'train_pairs':len(pairs),'schedule_sha256':schedhash,
        'initial_sha256':digest(initial),'checkpoint_sha256':digest(out/'checkpoint.pt'),'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters())}
    write_json(out/'complete.json',complete);write_json(out/'metrics.json',{'updates':256,'final_loss':item['loss']})

def dispatch_check(pin):
    require(digest(ROOT/'dispatch.json')==pin,'dispatch pin');d=read_json(ROOT/'dispatch.json');require(d['freeze']==digest(ROOT/'freeze.json'),'freeze drift')
    for n,h in d['recipes'].items():require(digest(ROOT/n)==h,'recipe drift')
    check()

def fit_all(pin):
    dispatch_check(pin);receipts=[]
    for si,seed in enumerate(SEEDS):
        for arm in ARMS if si%2==0 else tuple(reversed(ARMS)):
            for stage in STAGES if si%2==0 else tuple(reversed(STAGES)):
                dispatch_check(pin);name=f'{arm}-{seed}-{stage}';dest=ROOT/'runs'/name;require(not dest.exists(),'no silent retries')
                receipt=run_experiment(ROOT/(name+'.json'),dest);require(receipt['status']=='succeeded','fit failed retain attempt')
                path=dest/receipt['id'];require(verify_run(path)['valid'],'fit receipt');complete=read_json(path/'output/complete.json')
                receipts.append({'name':name,'run_id':receipt['id'],'receipt_sha256':digest(path/'run.json'),**complete});print('CLOSED',name,complete['seconds'],flush=True)
    for seed in SEEDS:require(len({r['schedule_sha256'] for r in receipts if r['seed']==seed})==1,'schedule mismatch')
    for arm in ARMS:
        for seed in SEEDS:require(len({r['initial_sha256'] for r in receipts if r['arm']==arm and r['seed']==seed})==1,'initial mismatch')
    write_json(ROOT/'training_closed.json',{'runs':receipts,'dispatch_sha256':pin,'all_closed_before_validation':True})
    return {'closed_fits':len(receipts)}

def weight_pin(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def aggregate(records):
    groups=defaultdict(list)
    for r in records:groups[r['world_id']].append(r['correct'])
    return {'correct':sum(r['correct'] for r in records),'total':len(records),'accuracy':sum(r['correct'] for r in records)/len(records),
        'complete_maps':sum(len(a)==4 and all(a) for a in groups.values()),'maps':len(groups),'map_accuracy':sum(len(a)==4 and all(a) for a in groups.values())/len(groups)}

def evaluate():
    check();closure=read_json(ROOT/'training_closed.json');dispatch_check(closure['dispatch_sha256']);require(len(closure['runs'])==12 and closure['all_closed_before_validation'],'incompleteclosure')
    out=ROOT/'results';out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pairs=[r['features']['pair'] for r in ds().examples('validation',cutoff=CUTOFF,feature_fields=['pair'])]
    rows=[]
    for p in pairs:
        for wi,w in enumerate(p['worlds']):
            require(history_map(w)==replay(w),'validation oracle')
            for q in KEYS:rows.append({**example(w,q),'world_id':p['pair_id']+f'-w{wi}','pair_id':p['pair_id'],'world':wi,'query':q,'root':p['root'],'new':p['new'],'template':p['template'],'focal_op':p['focal_op'],
                'role':next(k for k in ('root','alias','spare','unrelated') if p[k]==q)})
    sources=[]
    for seed in SEEDS:
        for arm in ARMS:sources.append({'arm':arm,'seed':seed,'stage':'starting','checkpoint':ROOT/'initial'/f'{arm}-{seed}.pt'})
    for r in closure['runs']:
        p=ROOT/'runs'/r['name']/r['run_id'];require(verify_run(p)['valid'] and digest(p/'run.json')==r['receipt_sha256'],'closedrun drift')
        require(digest(p/'output/checkpoint.pt')==r['checkpoint_sha256'],'closedcheckpoint drift');sources.append({**r,'checkpoint':p/'output/checkpoint.pt'})
    records=[];probes=[];vectors=[];summaries=[];weights=[];eval_started=time.perf_counter()
    with torch.inference_mode():
        for s in sources:
            model=build(s['arm']);model.load_state_dict(torch.load(s['checkpoint'],map_location='cpu',weights_only=True)['model']);model.eval();before=weight_pin(model)
            for condition in ('normal','erase_values'):
                part=[]
                for start in range(0,len(rows),64):
                    chunk=rows[start:start+64];b=tensors(chunk);roots=torch.tensor([int(r['root'][1:]) for r in chunk])
                    logits,_,state=forward_with_root(model,b,roots,mutation=None if condition=='normal' else 'erase_values')
                    require(bool(torch.isfinite(logits).all()),'nonfiniteeval')
                    for i,r in enumerate(chunk):
                        rec={k:r[k] for k in ('world_id','pair_id','world','query','root','new','template','focal_op','role','answer')}
                        rec.update(arm=s['arm'],seed=s['seed'],stage=s['stage'],condition=condition,prediction=int(logits[i].argmax()),correct=int(logits[i].argmax())==r['answer'],query_preserved=bool(model.last_preservation[i]),logits=logits[i].tolist())
                        if condition=='normal':rec['trace_index']=len(vectors);vectors.append(state[i].numpy())
                        records.append(rec);part.append(rec)
                summaries.append({'arm':s['arm'],'seed':s['seed'],'stage':s['stage'],'condition':condition,'overall':aggregate(part),
                    'templates':{t:aggregate([r for r in part if r['template']==t]) for t in TEMPLATES},
                    'template_focal':{t:{op:aggregate([r for r in part if r['template']==t and r['focal_op']==op]) for op in ('copy','link')} for t in TEMPLATES},
                    'preserved':sum(r['query_preserved'] for r in part)})
            grid=probe_generate();require(len({c['case_id'] for c in grid})==288,'probe identity')
            for start in range(0,len(grid),64):
                cs=grid[start:start+64];b=tensors([example(c['memory'],c['key']) for c in cs]);logits,_=model.sequence(b['ids'])
                for i,c in enumerate(cs):probes.append({'arm':s['arm'],'seed':s['seed'],'stage':s['stage'],'case_id':c['case_id'],'key':c['key'],'old':c['old'],'new':c['new'],'panel':c['panel'],'prediction':TOKENS[int(logits[i].argmax())],'correct':int(logits[i].argmax())==VOCAB[c['new']]})
            after=weight_pin(model);require(before==after,'evaluation mutated weights');weights.append({'arm':s['arm'],'seed':s['seed'],'stage':s['stage'],'before':before,'after':after,'checkpoint_sha256':digest(s['checkpoint'])})
            print('EVALUATED',s['arm'],s['seed'],s['stage'],flush=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records));(out/'probes.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in probes))
    np.savez_compressed(out/'root_states.npz',states=np.array(vectors));write_json(out/'summary.json',{'summaries':summaries,'validation_pairs':len(pairs),'closure_sha256':digest(ROOT/'training_closed.json'),'predictions_sha256':digest(out/'predictions.jsonl'),'probe_sha256':digest(out/'probes.jsonl')})
    write_json(out/'evaluation_receipt.json',{'closure_sha256':digest(ROOT/'training_closed.json'),'weights':weights,'seconds':time.perf_counter()-eval_started,'source_sha256':digest(ROOT/'experiment.py'),'validation_predictions':len(records),'probe_predictions':len(probes),'fits_closed_before_scoring':True})
    check();return {'validation_predictions':len(records),'probe_predictions':len(probes),'new_fits':12}

def register():
    f=check();out=ROOT/'results';closure=read_json(ROOT/'training_closed.json');report=json.loads((out/'analysis.json').read_text());lab=Controller(ROOT/'research_lab');manifest=f['manifest_sha256']
    import_forge(lab,ROOT/'dataset',manifest)
    for r in closure['runs']:import_trainlab(lab,ROOT/'runs'/r['name']/r['run_id'])
    ev=EvidenceRecord(evidence_type='research.set_invariance_continuation',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,
        claim='Matched continuation objective intervention on known-template DEV; not architecture qualification.',subjects=(EvidenceSubjectRef(kind='dataset',subject_id='set-repair-001',content_sha256=manifest),),
        producer=EvidenceProducerRef(producer_id='veritas-lifecycle-semantics',producer_version='1',content_sha256=_runtime()['source_trees']['investigation_world']),
        artifacts=(EvidenceArtifactRef(artifact_id='repair-predictions',content_sha256=digest(out/'predictions.jsonl')),),observed_at=datetime.now().astimezone(),provenance={'new_fits':12,'scope':'DEV_OBJECTIVE_INTERVENTION'})
    write_json(out/'evidence.json',ev.model_dump(mode='json'));put_node(lab,'set-repair-001','finding',report);put_node(lab,'set-repair-evidence-001','evidence',ev.model_dump(mode='json'))
    for target in ['set-repair-evidence-001','forge.'+manifest[:24]]+['trainlab.'+r['run_id'] for r in closure['runs']]:link(lab,'set-repair-001','derived_from',target)
    write_json(out/'knowledge.json',retrieve(lab));return {'registered_fits':12,'evidence_sha256':digest(out/'evidence.json')}

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':r=prepare()
    elif mode=='train':r=train(sys.argv[2],int(sys.argv[3]),sys.argv[4])
    elif mode=='fit':r=fit_all(sys.argv[2])
    elif mode=='evaluate':r=evaluate()
    elif mode=='register':r=register()
    else:raise ValueError('unknownmode')
    if r is not None:print(json.dumps(r,indent=2))
