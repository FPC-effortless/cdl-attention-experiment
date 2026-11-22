"""Prepare a label-free no-fit witness, requiring independent admission."""
from hashlib import sha256
import json,shutil
from pathlib import Path
from trainlab.runner import verify_run
R=Path(__file__).resolve().parent
BASE=R.parent/'cog-role-001'
B2=R.parent/'cog-role-002'
ACTOR=BASE/'panels/fit_actor.json'
A='7776cd8e328a32043253ffcc463a60fb97037ceb694eb23d7df0632ce2aaac4b'
L='2f28213c2eedcbdfce2151a42250ba57b0ca48bd92a2119726dc10818679ad25'
S={1171:'b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c',
   1371:'517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'}
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,allow_nan=False,indent=2)+'\n')
def main():
    ledger=B2/'actor-admissions/ledger.json'
    assert h(ledger)==L and h(ACTOR)==A
    old=read(ledger)
    out=R/'rank_admissions';out.mkdir(exist_ok=False)
    z={}
    for seed,cp in S.items():
        name=f'rank-{seed}'
        prior=f'w025-{seed}-1280';record=old[prior]
        assert record['checkpoint_sha256']==cp
        old_actor=B2/'actor-admissions'/prior
        run=B2/'fits'/record['fit_id']
        assert verify_run(run)['valid'] and h(run/'run.json')==record['fit_run_sha']
        assert read(run/'run.json')['artifacts']['checkpoint-1280.pt']==h(old_actor/'checkpoint.pt')==cp
        dest=out/name;dest.mkdir()
        files=[p for p in record['inputs'] if p.startswith(('cog_comm/','nanocog/')) and not p.endswith(('role_actor.py','role_judge.py','role_control.py'))]
        for p in files:
            assert h(old_actor/p)==record['inputs'][p]
            t=dest/p;t.parent.mkdir(exist_ok=True);shutil.copyfile(old_actor/p,t)
        for f in ('tap_gate_residual.py','frozen_gate_adapter.py','rank_geometry_worker.py','REPAIR_PROPOSAL.md'):
            shutil.copyfile(R/f,dest/f);files.append(f)
        shutil.copyfile(ACTOR,dest/'actor.json');shutil.copyfile(old_actor/'checkpoint.pt',dest/'checkpoint.pt')
        files+=['actor.json','checkpoint.pt']
        spec=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
                name=name,description='No-fit adapter parity and train prompt feature-rank input capture.',
                command=['python','rank_geometry_worker.py',cp,str(seed)],seed=seed,timeout_seconds=45,
                source_kind='CONSTRUCTED',split_identity='ROLE-REPAIR004-exact-train-prompt-projection',
                inputs=files,artifacts=['FEATURES.f32le','geometry.json'],
                metrics={'model_forwards':'min','optimizer_updates':'min','feature_rows_per_arm':'max'},
                budget={'model_forwards':664})
        write(dest/'recipe.json',spec)
        z[name]=dict(seed=seed,checkpoint_sha256=cp,actor_sha256=A,fit_id=record['fit_id'],
                    fit_run_sha=record['fit_run_sha'],recipe_sha256=h(dest/'recipe.json'),inputs={p:h(dest/p) for p in files})
    write(out/'ledger.json',z)
    print(h(out/'ledger.json'))
if __name__=='__main__':main()
