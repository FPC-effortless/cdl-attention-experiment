"""Synthetic temporary-file tests of closure validation, no real model receipts."""
import copy,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('slot_closure_review',ROOT/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def check(c,msg):
    if not c:raise RuntimeError(msg)
def run():
    outcomes={}
    for case in ['valid','duplicates','missing','early','receipt_mismatch','budget','freeze','checkpoint','log','steps','schedule']:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);orig=root/'original';m.ROOT=root;m.verify=lambda source:orig
            (root/'freeze.json').write_text('fixture only')
            receipts=[]
            for seed in [101,202,303]:
                d=root/'runs'/str(seed);d.mkdir(parents=True)
                (d/'checkpoint.pt').write_bytes(b'not a model: generated fixture')
                (d/'updates.jsonl').write_text('\n'.join(json.dumps({'step':i}) for i in range(1,257)))
                r={'seed':seed,'updates':256,'answer_exposures':8192,'freeze_sha256':m.sha(root/'freeze.json'),
                   'checkpoint_sha256':m.sha(d/'checkpoint.pt'),'training_log_sha256':m.sha(d/'updates.jsonl'),
                   'schedule_sha256':f'fixture{seed}'}
                dump(d/'complete.json',r);receipts.append(r)
                dump(orig/'runs'/f'supervised_{seed}'/'complete.json',{'schedule_sha256':f'fixture{seed}'})
            closure={'runs':receipts,'all_three_closed_before_stress_scoring':True}
            d=root/'runs'/'101';r=receipts[0]
            if case=='duplicates':closure['runs']=[receipts[0]]*3
            if case=='missing':closure['runs']=receipts[:2]
            if case=='early':closure['all_three_closed_before_stress_scoring']=False
            if case=='receipt_mismatch':r['additional']='x'
            if case=='budget':r['updates']=255;dump(d/'complete.json',r)
            if case=='freeze':r['freeze_sha256']='wrong';dump(d/'complete.json',r)
            if case=='checkpoint':(d/'checkpoint.pt').write_bytes(b'changed')
            if case=='log':(d/'updates.jsonl').write_text('{}')
            if case=='steps':
                (d/'updates.jsonl').write_text('\n'.join(json.dumps({'step':i}) for i in range(256,0,-1)))
                r['training_log_sha256']=m.sha(d/'updates.jsonl');dump(d/'complete.json',r)
            if case=='schedule':r['schedule_sha256']='wrong';dump(d/'complete.json',r)
            dump(root/'training_closed.json',closure)
            try:
                m.validate_closure(root)
                outcomes[case]='ACCEPT'
            except RuntimeError:outcomes[case]='REJECT'
            check(outcomes[case]==('ACCEPT' if case=='valid' else 'REJECT'),f'Unexpected {case}')
    return {'status':'PASS','cases':outcomes,'source_sha256':hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest(),
            'scope':'Generated temporary receipts. verify() mocked only to isolate closure logic; no genuine training claim or neural execution.'}
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
