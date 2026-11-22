"""Native receipt, trace and adapter-tensor replay; no model execution or fit."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean,median
from itertools import product
import json,math,random,sys
import torch
from trainlab.runner import verify_run
from trainlab.research_adequacy import recipe_identity
from trainlab.execution_adequacy import assess_execution

R=Path(__file__).resolve().parent
LED='290ae7e56763ccba99d36fe3513fd2c0f34a3e98eebac58f33547c1b17aee68a'
SEAL=sys.argv[1]
BASE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
TRAIN='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8'
PROTOCOL='b0ba8b65677ccb8a29c4fb2992321247b0099a72acad350709a8a03a4ffa2889'
SEEDS=(1771,1871,1971)
SCHEDULE={1771:'0baad73a727e7fd98d528f5d576c20360fc671f25cc2832d76ba0d2311906f54',
1871:'992e71eca26bb023cd9df14171f2d3302e1b7e99b724336572097069ba6bb2aa',
1971:'8790854f2b555ab2247357c0a26362343199d56de59ed0d5a6cd99a2ce15d715'}
ARMS=('consistency000','consistency025')
WEIGHT={'consistency000':0.,'consistency025':.25}
def need(ok,h):
    if not ok:raise ValueError(h)
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
    b=p.read_bytes();need(pin is None or sha256(b).hexdigest()==pin,f'hash {p}')
    return json.loads(b)
def tensor_sha(tensors):
    h=sha256()
    for n,t in sorted(tensors.items()):
        need(type(n)==str and torch.is_tensor(t) and t.device.type=='cpu' and t.dtype==torch.float32
          and bool(torch.isfinite(t).all()),'adapter/base tensor geometry')
        h.update(n.encode());h.update(str((list(t.shape),str(t.dtype))).encode());h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()
def pure_rng_hash(seed):
    # Reconstruct the initialization directly; no module or training invocation.
    g=torch.Generator(device='cpu').manual_seed(seed)
    first=torch.empty(8,68,dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(first,a=math.sqrt(5),generator=g)
    return tensor_sha({'first.weight':first,'last.weight':torch.zeros(1,8)})
d=R/'fit_admissions';w=R/'fit_runs'
ledger=read(d/'ledger.json',LED);seal=read(R/'CONSIST_FIT_SEAL.json',SEAL)
need(seal['ledger_sha256']==LED and set(seal['runs'])==set(ledger)=={f'{a}-{s}' for a,s in product(ARMS,SEEDS)},'fit roster')
old=R.parent/'cog-role-002'
oldmeta=read(old/'ROLE002_FIT_INDEPENDENT_AUDIT.json','92d9bbf3b1b4643236be235fa7c9292dcceb29de7aeb65eca5a00ce3e2cf880c')['runs']['1371']
oldrun=old/'fits'/oldmeta['run_id'];base_file=oldrun/'output/checkpoint-1280.pt'
need(verify_run(oldrun)['valid'] and sha(oldrun/'run.json')==oldmeta['run_sha256']
   and sha(base_file)==BASE and oldmeta['endpoints']['1280']['checkpoint_sha256']==BASE,'frozen upstream')
base=torch.load(base_file,map_location='cpu',weights_only=True)
base_state=tensor_sha(base['state_dict'])
need(base_state==oldmeta['endpoints']['1280']['state_sha256'] and sum(x.numel() for x in base['state_dict'].values())==37414,'base tensor lineage')
out={}; seen_ck=set()
kn=('adapter-128.pt','adapter-1280.pt','fit.json','attempt.json','exposure.json','steps.jsonl','observations.json')
for key,pin in ledger.items():
    arm,seed=pin['arm'],pin['seed'];case=d/key; e=seal['runs'][key]; run=w/e['id']; p=run/'output'
    r=read(run/'run.json',e['receipt_sha256']); recipe=read(case/'recipe.json',pin['recipe_sha256'])
    need(r==read(case/'cli-output.json') and verify_run(run)['valid'] and r['name']==key and r['seed']==seed
       and r['metrics']=={'committed_updates':1280,'model_forwards':1283} and 0<r['elapsed_seconds']<=180
       and r['execution_purpose']=='research' and r['research_adequacy']=='EXECUTED_CRITERIA_MET'
       and r['qualification']=='NOT_QUALIFIED' and r['status']=='succeeded' and r['exit_code']==0,'native run integrity')
    need(recipe==read(run/'experiment.json') and r['inputs']==pin['inputs']
      and r['artifacts']==e['artifacts'] and set(r['artifacts'])==set(kn) and all(sha(p/n)==h for n,h in e['artifacts'].items())
      and all(sha(case/n)==h for n,h in r['inputs'].items())
      and r['inputs']['train.json']==TRAIN and r['inputs']['checkpoint.pt']==BASE
      and r['inputs']['NAME_CONSIST_EVAL_LOCK.md']==PROTOCOL
      and not any('actor.json' in n or 'judge' in n or 'holdout' in n for n in r['inputs']),'byte-bound native closure')
    con=read(case/'adequacy.json',r['inputs']['adequacy.json'])
    ob=read(p/'observations.json',e['artifacts']['observations.json'])
    need(ob['inputs']==r['inputs'] and ob['producer']=={'file':'fit_name_consistency.py','sha256':r['inputs']['fit_name_consistency.py']}
      and ob['recipe_sha256']==recipe_identity(recipe)==pin['recipe_identity']
      and assess_execution(con,recipe,r['inputs'],ob)['status']=='MET','postfit admission')
    f=read(p/'fit.json'); at=read(p/'attempt.json'); pl=f['plan']; coeff=WEIGHT[arm]
    need(f['consistency_weight']==at['consistency_weight']==pin['coefficient']==coeff,'registered coefficient')
    need(pl==at['plan']==dict(seed=seed,arm=arm,backbone_seed=1371,steps=1280,saves=[128,1280],schedule_seed_xor=0xC060001,
        lr=.001,warmup=8,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip=1.,alpha=1.,prefix_ce_weight=0.)
      and f['backbone_state_sha256']==at['backbone_state_sha256']==base_state
      and f['backbone_sha256']==at['backbone_sha256']==BASE
      and f['adapter_initial_sha256']==at['adapter_initial_sha256']==pure_rng_hash(seed)
      and f['torch']==at['torch']==torch.__version__=='2.10.0+cpu'
      and at['intraop_threads']==at['interop_threads']==1 and at['dtype']=='float32'
      and at['executable_sha256']==sha(Path(sys.executable).resolve()),'initialization/runtime')
    sc=read(case/'schedule.json',SCHEDULE[seed]); ex=read(p/'exposure.json');rng=random.Random(seed^0xC060001); order=[]
    for _ in range(4):
        t=list(range(320));rng.shuffle(t);order+=t
    need(sc['schema']=='cog-role-name-consist.schedule.v1' and sc['seed']==seed and sc['train_sha256']==TRAIN and
         f['schedule_sha256']==at['schedule_sha256']==ex['source_sha256']==pin['schedule_sha256']==SCHEDULE[seed] and
         len(sc['steps'])==1280,'schedule lineage')
    original=read(case/'train.json',TRAIN)['families']; counts=Counter(); edges={}; observed=set(); prim=set(); s_counts=Counter()
    for i,(family,stage) in enumerate(zip(order,sc['steps'],strict=True)):
        refs=stage['rows']; rows=[original[k]['prompts'][j].split() for k,j in refs]
        y=[original[k]['answers'][j] for k,j in refs]
        need(stage['primary']==family and len(set(map(tuple,refs)))==8 and len(rows)==len(y)==8,'raw row roster')
        for a,b in ((0,1),(2,3),(4,5),(6,7)):
            need(refs[a][0]==refs[b][0] and rows[a][2]==rows[b][5] and rows[a][5]==rows[b][2] and
                all(rows[a][k]==rows[b][k] for k in (0,1,3,4,6)) and y[a]!=y[b],'raw foil replay')
        if stage['kind']=='ordinary':
            need(refs==[[family,n] for n in (0,4,1,5,2,6,3,7)],'ordinary original family')
        else:
            need(stage['kind']=='paired' and [x[0] for x in refs]==[family,family,refs[2][0],refs[2][0]]*2
                 and refs[2][0]!=family,'paired original families')
            edge=tuple(sorted({family,refs[2][0]}))
            need(len(edge)==2 and (stage['edge'] not in edges or edges[stage['edge']]==edge),'remap edge identity')
            edges[stage['edge']]=edge
            for a,b in ((0,2),(1,3),(4,6),(5,7)):
                typ=stage['strata'][a//4]; edit={i for i in range(7) if rows[a][i]!=rows[b][i]}
                query_first=rows[a][0]==rows[a][-1]; loc=0 if typ.endswith('first') else 3
                want={loc,6} if typ.startswith('focal') else {3-loc}
                need(edit==want and query_first==(loc==0) and y[a]==y[b] and y[a^1]==y[b^1]
                     and (6 not in edit or (rows[a][loc],rows[b][loc])==(rows[a][6],rows[b][6])),'exact remap replay')
            s_counts.update(stage['strata'])
        counts[stage['kind']]+=1; observed.update(map(tuple,refs));prim.update(tuple(v) for v in refs if v[0]==family)
    need(counts=={'paired':1188,'ordinary':92} and len(edges)==len(set(edges.values()))==444 and
      len(observed)==len(prim)==2560 and ex['edge_count']==444 and ex['served_original_worlds']==2560 and
      ex['ordinary_family_count']==23 and ex['served_original_families']==320 and ex['stratum_counts']==dict(s_counts),'matched training exposures')
    tr=[json.loads(row) for row in (p/'steps.jsonl').read_bytes().splitlines()]
    need(len(tr)==1280 and f['trace_sha256']==sha(p/'steps.jsonl'),'trace roster')
    for i,t in enumerate(tr,1):
        gr=t['adapter_nonzero_gradient_counts']; paired=sc['steps'][i-1]['kind']=='paired'
        need(0<=t['remap_consistency']<=1 and (paired or t['remap_consistency']==0.) and
             t['is_paired']==int(paired) and t['consistency_weight']==coeff,'actual bounded consistency')
        need(t['step']==i and t['family_ordinal']==order[i-1] and t['lr']==.001*min(1,i/8)
          and [t[k] for k in ('train_worlds','train_input_tokens','answer_targets','paired_foil_pairs','prefix_targets')]==[8,56,8,4,0]
          and math.isfinite(t['raw_gradient_norm']) and t['raw_gradient_norm']>0
          and t['above_clip_threshold']==(t['raw_gradient_norm']>1.)
          and set(gr)=={'first.weight','last.weight'} and 0<=gr['first.weight']<=544 and 0<=gr['last.weight']<=8
          and all(math.isfinite(t[k]) for k in ('objective','answer_ce_nats','paired_foil_softplus','answer_minus_foil_mean'))
          and abs(t['objective']-t['answer_ce_nats']-t['paired_foil_softplus']-coeff*t['remap_consistency'])<3e-6,'trace loss/schedule/gradient')
    need(tr[0]['adapter_nonzero_gradient_counts']['first.weight']==0 and tr[0]['adapter_nonzero_gradient_counts']['last.weight']>0
      and any(t['adapter_nonzero_gradient_counts']['first.weight']>0 for t in tr[1:]),'zero-start gradient behavior')
    ends={}
    for n in (128,1280):
        m=f['checkpoints'][str(n)];ap=p/f'adapter-{n}.pt';pkt=torch.load(ap,map_location='cpu',weights_only=True)
        expected={'schema','format','plan','arm','seed','updates','backbone_seed','backbone_sha256','backbone_state_sha256',
          'adapter_initial_sha256','adapter_state_sha256','train_sha256','schedule_sha256','tokenizer_words','vocab_size','state_dict','model_arm','consistency_weight'}
        need(set(pkt)==expected and pkt['schema']=='cog-role-name-consist-001.adapter.v1' and pkt['model_arm']=='tap_compare' and pkt['consistency_weight']==coeff and pkt['format']==1
          and pkt['seed']==seed and pkt['arm']==arm and pkt['updates']==n and pkt['backbone_seed']==1371 and pkt['train_sha256']==TRAIN
          and pkt['plan']==pl|{'betas':tuple(pl['betas']),'saves':tuple(pl['saves'])}
          and pkt['backbone_sha256']==BASE and pkt['backbone_state_sha256']==base_state and pkt['vocab_size']==287
          and pkt['schedule_sha256']==f['schedule_sha256'] and pkt['tokenizer_words']==base['tokenizer_words']
          and pkt['adapter_initial_sha256']==f['adapter_initial_sha256'],'endpoint metadata')
        state=pkt['state_dict']; st=tensor_sha(state); af=sha(ap)
        need({k:tuple(v.shape) for k,v in state.items()}=={'first.weight':(8,68),'last.weight':(1,8)}
          and st==m['adapter_state_sha256']==pkt['adapter_state_sha256']!=pkt['adapter_initial_sha256']
          and af==m['sha256']==r['artifacts'][ap.name] and m['artifact']==ap.name and af not in seen_ck,'adapter-only wire')
        seen_ck.add(af)
        co=m['counters']; ni=2 if n==128 else 3
        match=sum(t['is_paired'] for t in tr[:n])
        need(co==dict(model_forwards=n+ni,training_forwards=n,integrity_forwards=ni,committed_updates=n,backwards=n,
            train_worlds=8*n,train_input_tokens=56*n,answer_targets=8*n,paired_foil_pairs=4*n,
            paired_updates=match,ordinary_updates=n-match,consistency_pairs=4*match)
            and m['unique_families']==len(set(order[:n])) and m['state_parameters']==552,'endpoint exposure')
        for a,b in [('objective','objective'),('answer_ce','answer_ce_nats'),('foil','paired_foil_softplus'),('consistency','remap_consistency'),('raw_gradient_norm','raw_gradient_norm')]:
            need(math.isclose(m['mean_trace'][a],mean(t[b] for t in tr[:n]),rel_tol=1e-13,abs_tol=1e-14),'aggregate trace cross-check')
        ends[str(n)]={'sha256':af,'adapter_state_sha256':st,'families':m['unique_families']}
    need(ends['128']['adapter_state_sha256']!=ends['1280']['adapter_state_sha256']
      and f['counters']==ob['observations'] and f['status']=='COMPLETE'
      and f['counters']['frozen_base_integrity_checks']==f['counters']['earlier_logits_endpoint_parity_checks']==2,'terminal native counters')
    out[key]=dict(run_id=run.name,trace_rows=len(tr),elapsed_seconds=r['elapsed_seconds'],schedule_sha256=f['schedule_sha256'],
      initial_adapter_sha256=f['adapter_initial_sha256'],trace_sha256=f['trace_sha256'],checkpoints=ends,
      first_step={n:tr[0][n] for n in ('answer_ce_nats','paired_foil_softplus','answer_minus_foil_mean','remap_consistency')},
      consistency_weight=coeff,planned_pairs=4752,edge_count=444,
      max_loss_formula_error=max(abs(t['objective']-t['answer_ce_nats']-t['paired_foil_softplus']-coeff*t['remap_consistency']) for t in tr),
      mean_consistency_on_paired_updates=mean(t['remap_consistency'] for t in tr if t['is_paired']),
      clipping_count=sum(t['above_clip_threshold'] for t in tr),
      median_raw_gradient=median(t['raw_gradient_norm'] for t in tr))
for s in SEEDS:
    x,y=[out[f'{a}-{s}'] for a in ARMS]
    need(all(x[k]==y[k] for k in ('schedule_sha256','initial_adapter_sha256','first_step'))
      and all(x['checkpoints'][n]['adapter_state_sha256']!=y['checkpoints'][n]['adapter_state_sha256'] for n in ('128','1280')),
      'paired seeds mismatch')
need(len(seen_ck)==12 and len({out[f'consistency025-{s}']['initial_adapter_sha256'] for s in SEEDS})==3,'paired roster')
result=R/'INDEPENDENT_CONSIST_FIT_AUDIT.json'
with result.open('x') as f:json.dump(dict(seal_sha256=SEAL,base_sha256=BASE,base_state_sha256=base_state,
    model_invocations=0,fit_updates_by_auditor=0,runs=out),f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(sha(result),sum(v['elapsed_seconds'] for v in out.values()),'s')
for n,v in out.items():print(n,v['run_id'],v['clipping_count'],v['median_raw_gradient'])
