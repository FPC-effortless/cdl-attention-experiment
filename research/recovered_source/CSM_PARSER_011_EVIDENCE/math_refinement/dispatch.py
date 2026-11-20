from pathlib import Path
import json,hashlib,sys,platform
from forge_data.core import release,verify_release
from trainlab.runner import run_experiment,verify_run
from research_lab.core import Controller
from research_lab.evidence import put_node
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=B/'cases.jsonl';raw.write_text(json.dumps({'episode_id':'exhaustive-3','subject_id':'finite-state','record_id':'1','observed_at':'2026-11-16T00:00:00Z','available_at':'2026-11-16T00:00:00Z','observation':{'codes':list(range(5)),'actions':list(range(5)),'forced_action_diagnostic':True,'exhaustive_lengths':[1,2,3,4,5],'random_lengths':[8,64,257,1024],'random_seed':111011},'outcome':None})+'\n')
spec=B/'forge_spec.json';spec.write_text(json.dumps({'dataset_id':'csm-parser-011-math','source_id':'constructed-exhaustive','source_version':'1','license_id':'owner-created','provenance_uri':'local:verify.py','ai_use_allowed':True,'redistribution_allowed':True,'privacy_reviewed':True,'adapter':'operations-jsonl','split_salt':'no-training-executor-only'}))
release(raw,spec,B/'forge_release');(B/'forge_integrity.json').write_text(json.dumps(verify_release(B/'forge_release'),indent=2))
(B/'runtime.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'lean_sha256':sha(Path('/tmp/lean-4.15.0-linux/bin/lean')),'lean_version':'4.15.0','provenance':'https://github.com/leanprover/lean4/releases/download/v4.15.0/lean-4.15.0-linux.tar.zst'}))
inputs=['Carry.lean','check.py','driver.py','cases.jsonl','forge_spec.json','runtime.json','study_frozen.py']
config={'schema':'trainlab.experiment.v1','name':'csm-parser-011-math','description':'Carrier list-fold refinement and forced-action helper agreement, zero learned models','execution_purpose':'engineering_diagnostic','command':[str(B.parent/'runtime_recovery/cpu-venv/bin/python'),'driver.py'],'seed':0,'timeout_seconds':120,'source_kind':'CONSTRUCTED','split_identity':sha(raw),'inputs':inputs,'artifacts':['RESULTS.json','metrics.json'],'metrics':{'cases':'max'},'budget':{'model_fits':0}}
(B/'recipe.json').write_text(json.dumps(config,indent=2))
r=run_experiment(B/'recipe.json',B/'runs');(B/'receipt.json').write_text(json.dumps(r,indent=2));v=verify_run(B/'runs'/r['id']);(B/'integrity.json').write_text(json.dumps(v,indent=2))
lab=Controller(B/'research_lab');put_node(lab,'csm-parser-011-math','evidence',{'receipt':r,'integrity':v,'scope':'executor engineering only; no model qualification'})
print(r['status'],v)
assert r['status']=='succeeded' and v['valid']
