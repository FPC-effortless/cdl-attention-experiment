"""Stage two exact no-fit prediction closures; does not launch."""
import json,shutil
from hashlib import sha256
from pathlib import Path
from trainlab.runner import verify_run
P=Path(__file__).resolve().parent
R=P.parent/'cog-role-002'
A='44f380f0e934450e3c809162578381e670683602a4e52a4487c89fd2197a8116'
S={'1171':'b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c',
   '1371':'517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'}
def sh(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,allow_nan=False,sort_keys=True,indent=2)+'\n')
H=P/'probe_admissions';H.mkdir(exist_ok=False)
m={}
for seed,cp in S.items():
 base=R/'actor-admissions'/f'w025-{seed}-1280'
 reg=R/'actor-admissions/ledger.json'
 assert sh(reg)=='2f28213c2eedcbdfce2151a42250ba57b0ca48bd92a2119726dc10818679ad25'
 old=read(reg)[base.name]
 fit=R/'fits'/old['fit_id']
 assert verify_run(fit)['valid'] and sh(fit/'run.json')==old['fit_run_sha']
 assert cp==old['checkpoint_sha256']==sh(base/'checkpoint.pt')==read(fit/'run.json')['artifacts']['checkpoint-1280.pt']
 dest=H/f'probe-{seed}';dest.mkdir()
 files=[z for z in old['inputs'] if z.startswith(('cog_comm/','nanocog/')) and not z.endswith(('role_actor.py','role_judge.py','role_control.py'))]
 for rel in files:
  assert sh(base/rel)==old['inputs'][rel]
  (dest/rel).parent.mkdir(exist_ok=True);shutil.copyfile(base/rel,dest/rel)
 assert sh(P/'probe_panel/actor.json')==A
 shutil.copyfile(P/'probe_panel/actor.json',dest/'actor.json')
 shutil.copyfile(base/'checkpoint.pt',dest/'checkpoint.pt')
 for x in ('query_gate_probe.py','run_gate_probe.py','GATE_PROBE_PROTOCOL.md'):
  shutil.copyfile(P/x,dest/x)
  files.append(x)
 files+=['actor.json','checkpoint.pt']
 d=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
        name=dest.name,description='Frozen no-fit second-layer gate diagnostic',
        command=['python','run_gate_probe.py',A,cp,seed],seed=int(seed),timeout_seconds=45,
        source_kind='CONSTRUCTED',split_identity='ROLE-GATE-001-fourth-fresh-synthetic-actor',inputs=files,
        artifacts=['predictions.json'],metrics={'prediction_rows':'max','model_forwards':'min','optimizer_updates':'min'},
        budget={'inference_batches':160})
 write(dest/'recipe.json',d)
 m[dest.name]=dict(seed=int(seed),checkpoint_sha256=cp,actor_sha256=A,fit_id=old['fit_id'],
  fit_run_sha=old['fit_run_sha'],recipe_sha256=sh(dest/'recipe.json'),inputs={x:sh(dest/x) for x in files})
write(H/'ledger.json',m)
print(sh(H/'ledger.json'))
