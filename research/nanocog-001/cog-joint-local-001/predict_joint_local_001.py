"""One reviewed blind actor, no synthesis from held role/answer fields. Inert on import."""
import json,os,sys
from pathlib import Path
from joint_eval_checkpoint import decode_raw,state_sha,WORDS
from joint_local_intake import digest
from joint_local_objective import need

PANELS={'edits':768,'ordinary':960}
METRICS=dict(prediction_rows=1728,model_forwards=216,optimizer_updates=0)

def main():
    import torch
    root=Path.cwd(); out=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    need(out.is_dir() and not out.is_symlink() and not any(out.iterdir()) and not sys.flags.optimize and
         sys.version_info[:2]==(3,12) and torch.__version__=='2.10.0+cpu' and
         all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),
         'fresh exact CPU actor runtime')
    config=json.loads((root/'recipe.json').read_bytes()); pins=json.loads((root/'predict_inputs.json').read_bytes())
    core={n for n in pins if not n.startswith(('nanocog/','cog_comm/'))}
    need(core=={'checkpoint.pt','predict_joint_local_001.py','joint_eval_checkpoint.py','joint_local_intake.py',
                'joint_local_schedule.py','joint_local_objective.py','actor_case.json','JOINT_PROTOCOL_LOCK.md',
                'JOINT_EVAL_SOURCE_LOCK.md','actor-edits.json','actor-ordinary.json'} and
         set(config['inputs'])==set(pins)|{'predict_inputs.json'} and config['timeout_seconds']==45 and
         config['command']==['python','predict_joint_local_001.py'] and
         config['budget']=={'model_forwards':216,'optimizer_updates':0} and
         all((root/f).is_file() and not (root/f).is_symlink() and
             (f.startswith('actor-') or digest(root/f)==h) for f,h in pins.items()),
         'no TRAIN role labels or held truth admitted to the predictor')
    case=json.loads((root/'actor_case.json').read_bytes()); expected=case['checkpoint'];rt=case['runtime']
    need(config['seed']==expected['seed']==int(os.environ['TRAINLAB_SEED']) and
         digest(Path(sys.executable).resolve())==rt['python_sha256'] and
         digest(Path(torch.__file__).parent/'lib/libtorch_cpu.so')==rt['torch_cpu_sha256'] and
         expected['sha256']==pins['checkpoint.pt'],'exact case before model source, checkpoint, or rows')
    path=root/'checkpoint.pt'
    need(path.stat().st_size<=512_000,'bounded checkpoint bytes')
    mixer,sd=decode_raw(path.read_bytes(),expected,torch=torch)
    import cog_comm.role_mixers as roles
    import nanocog.model as backbone
    need(all(Path(m.__file__).is_relative_to(root) for m in (roles,backbone)),'exact decoder source')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    model=roles.build_arm(287,mixer,expected['seed']).eval().requires_grad_(False)
    model.load_state_dict(sd,strict=True)
    need(state_sha(model.state_dict())==expected['state_sha256'],'strict loaded-state identity')
    ids={word:262+i for i,word in enumerate(WORDS)}
    fw=0;cards={}
    with torch.inference_mode():
        for k,n in PANELS.items():
            key=f'actor-{k}.json';p=root/key
            need(p.stat().st_size<=120_000 and digest(p)==pins[key],'exact opaque prompt bytes')
            data=json.loads(p.read_bytes())
            need(set(data)=={'schema','prompts'} and data['schema']=='joint-local.actor.v1' and
                 type(data['prompts'])is list and len(data['prompts'])==len(set(data['prompts']))==n,
                 'opaque exact prompt roster')
            x=[]
            for raw in data['prompts']:
                a=raw.split()
                need(len(a)==7 and raw==' '.join(a) and all(w in ids for w in a),'seven visible literal words')
                x.append(tuple(ids[w] for w in a))
            file=out/f'{k.upper()}-LOGITS.f32le'
            with file.open('xb') as f:
                for i in range(0,n,8):
                    z=model(torch.tensor(x[i:i+8],dtype=torch.long),state=None)[0][:,6,:]
                    need(z.shape==(8,287) and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'exact finite logits')
                    f.write(z.contiguous().numpy().astype('<f4',copy=False).tobytes());fw+=1
            cards[k]=dict(shape=[n,287],actor_sha256=pins[key],logits_sha256=digest(file))
    need(fw==METRICS['model_forwards'] and state_sha(model.state_dict())==expected['state_sha256'],
         'fixed forward budget and unchanged model')
    with (out/'actor_receipt.json').open('x') as f:
        json.dump(dict(schema='cog-joint-local-001.blind-native.v1',arm=expected['arm'],seed=expected['seed'],
          updates=expected['updates'],mixer=mixer,checkpoint_sha256=expected['sha256'],
          state_sha256=expected['state_sha256'],tokenizer_words=list(WORDS),panels=cards,metrics=METRICS),
          f,sort_keys=True,allow_nan=False)
    with (out/'metrics.json').open('x') as f:json.dump(METRICS,f)

if __name__=='__main__':main()
