"""Label-free native value-repair actor; inert on import. No trained arrays in source."""
from hashlib import sha256
from io import BytesIO
import json,os,sys
from pathlib import Path

from load_value_adapter_v2 import safe_adapter,isolate_allowlist
from fit_value_path import BASE,BASE_TENSOR,OLD_TRAIN,ARMS,SEEDS,ENDPOINTS,KEYS,hash_file,need,save

PANELS={'edits':768,'ordinary':960}

def pinned(path,digest):
    need(path.is_file() and not path.is_symlink(),'regular staged input')
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,'changed actor/artifact '+path.name)
    return raw

def main():
    import torch
    root=Path.cwd()
    cfg=json.loads((root/'recipe.json').read_bytes()); closure=json.loads((root/'predict_inputs.json').read_bytes())
    need(set(cfg['inputs'])==set(closure)|{'predict_inputs.json'} and cfg['timeout_seconds']==45 and
         cfg['budget']=={'model_forwards':216,'optimizer_updates':0} and
         all(Path(k).is_file() and not Path(k).is_symlink() for k in closure),
         'exact native actor closure inventory')
    allowed={'actor-edits.json','actor-ordinary.json','actor_case.json','checkpoint.pt','fit_value_path.py',
             'frozen_value_port.py','value_path_residual.py','predict_value_path_v3.py','VALUE_EVAL_SOURCE_LOCK.md',
             'VALUE_EVAL_V2_COMPATIBILITY_LOCK.md','VALUE_EVAL_V3_DRY_LOCK.md','load_value_adapter_v2.py'}
    case=json.loads((root/'actor_case.json').read_bytes()); arm,seed,step=(case[k] for k in ('arm','seed','updates'))
    active=arm!='native'
    need(set(k for k in closure if not k.startswith(('nanocog/','cog_comm/')))==allowed|({'adapter.pt'} if active else set()),
         'actor is not label-free/or has extra inputs')
    need(((arm in ARMS and seed in SEEDS and step in ENDPOINTS) if active else (seed,step)==(1371,0)) and
         sys.version_info[:2]==(3,12) and not sys.flags.optimize and torch.__version__=='2.10.0+cpu' and
         closure['checkpoint.pt']==BASE and os.environ.get('TRAINLAB_SEED')==str(seed) and
         hash_file(Path(sys.executable).resolve())==case['python_sha256'] and
         hash_file(Path(torch.__file__).parent/'lib/libtorch_cpu.so')==case['torch_cpu_sha256'] and
         all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'frozen actor runtime')
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    need(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'fresh native output')
    need(cfg['command']==['python','predict_value_path_v3.py'] and cfg['seed']==seed and
         all(hash_file(Path(k))==v for k,v in closure.items()
             if k not in ('actor-edits.json','actor-ordinary.json','checkpoint.pt','adapter.pt')),
         'verified compatibility source and case before adapter')
    preloaded=None
    ad=None
    if active:
        ad=closure['adapter.pt']
        need(ad==case['adapter_sha256'],'actor adapter hash')
        expect=case['adapter_expectation']
        need((expect['arm'],expect['seed'],expect['updates'],expect['sha256'])==(arm,seed,step,ad),
             'scoped preload/init identity')
        # Do not import model packages, initialize devices/seeds, touch the
        # default weights-only base loader, or open prompts until this ends.
        original_globals=torch.serialization.get_safe_globals()
        isolate_allowlist()
        try:
            preloaded=safe_adapter(root/'adapter.pt',expect,case['torch_code_sha256'])
        finally:
            need(not torch.serialization.get_safe_globals(),'scoped adapter allowlist leaked')
            torch.serialization.add_safe_globals(original_globals)
        need(set(torch.serialization.get_safe_globals())==set(original_globals),'default Torch registry not restored')
    else:
        need(case['adapter_sha256'] is None and case['adapter_expectation'] is None,'native has zero adapter work')
    need(all(hash_file(Path(k))==v for k,v in closure.items()),'exact declared input bytes')
    import cog_comm.role_mixers as role
    import cog_comm.fit as oldfit
    import nanocog.model as native
    from frozen_value_port import FrozenValueRepairPort
    from value_path_residual import ValuePathResidual
    need(all(Path(m.__file__).is_relative_to(root) for m in (role,oldfit,native)),'frozen source origins')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    base=torch.load(BytesIO(pinned(root/'checkpoint.pt',BASE)),map_location='cpu',weights_only=True)
    need(base['schema']=='cog-comm-role-001.checkpoint.v1' and base['seed']==1371 and base['updates']==1280 and
         base['vocab_size']==287 and base['train_sha256']==OLD_TRAIN and base['plan']['arm']=='gated32' and
         base['plan']['clip']==1. and base['plan']['prefix_ce_weight']==.25,'base provenance')
    words=list(base['tokenizer_words'])
    need(len(words)==len(set(words))==25,'pinned word roster')
    ids={w:262+i for i,w in enumerate(words)}
    bb=role.build_arm(287,'gated32',1371).eval().requires_grad_(False)
    bb.load_state_dict(base['state_dict'],strict=True)
    digest=lambda m:oldfit.stable_tensor_hash(m.state_dict(),lambda _:True)
    need(digest(bb)==BASE_TENSOR and sum(x.numel() for x in bb.parameters())==37414,'frozen base tensor provenance')
    ad_state=None
    if active:
        a=preloaded
        need(a['tokenizer_words']==words,'same frozen word ids')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);head=ValuePathResidual(arm=arm)
        need(digest(head)==a['initial_adapter_state_sha256'],'adapter initialization provenance')
        head.load_state_dict(a['state_dict'],strict=True); ad_state=digest(head)
        need(ad_state==a['adapter_state_sha256'] and sum(x.numel() for x in head.parameters())==800 and
             all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in head.parameters()),'adapter tensor provenance')
        model=FrozenValueRepairPort(bb,head).eval().requires_grad_(False)
    forwards=0; panel_art={}
    with torch.inference_mode():
        for panel,n in PANELS.items():
            k='actor-'+panel+'.json'
            actor=json.loads(pinned(root/k,closure[k]))
            prompts=actor['prompts']
            need(set(actor)=={'schema','prompts'} and actor['schema']=='value-repair.blind-prompts.v1'
                and type(prompts)is list and len(prompts)==len(set(prompts))==n,'blind actor projection')
            tokens=[p.split() for p in prompts]
            need(all(len(w)==7 and ' '.join(w)==p for p,w in zip(prompts,tokens,strict=True)),'seven visible raw tokens only')
            buf=[tuple(ids[w] for w in sent) for sent in tokens]
            name=f'{panel.upper()}-LOGITS.f32le'
            with (out/name).open('xb') as f:
                for i in range(0,n,8):
                    x=torch.tensor(buf[i:i+8],dtype=torch.long)
                    z,_=model(x,enabled=True) if active else bb(x,state=None)
                    z=z[:,-1,:]
                    need(z.shape==(8,287) and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'full finite native logits')
                    f.write(z.contiguous().numpy().astype('<f4',copy=False).tobytes())
                    forwards+=1
            panel_art[panel]=dict(shape=[n,287],actor_sha256=closure[k],logits_sha256=hash_file(out/name))
    need(forwards==216 and digest(bb)==BASE_TENSOR and (not active or digest(head)==ad_state),'immutable weights and exact scope')
    m=dict(prediction_rows=1728,model_forwards=forwards,optimizer_updates=0)
    save(out/'actor_receipt.json',dict(schema='cog-info-value-repair.actor.v1',arm=arm,seed=seed,updates=step,
        backbone_sha256=BASE,adapter_sha256=ad,adapter_state_sha256=ad_state,tokenizer_words=words,panels=panel_art,
        metrics=m,native_zero_adapter_computation=not active))
    save(out/'metrics.json',m)

if __name__=='__main__':main()
