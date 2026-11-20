"""Two separate native diagnostic stages, held execution gated by exact reviews."""
import json,os,sys,shutil
from pathlib import Path
from trainlab.runner import run_experiment,verify_run,digest,load_config
from research_lab.core import Controller
from research_lab.evidence import put_node,link
BASE=Path(__file__).resolve().parent
FROZEN=BASE/'frozen'
E=BASE/'execution'
SOURCES=('codec.py','timeline.py','rules.py','primitive_reference.py','generate.py','worker.py')
PREFIT=(*SOURCES,'lab.py','PROTOCOL.md','test_clock.py','frozen/observations.json','frozen/private-reference.json','frozen/forge-receipt.json')
if sys.flags.optimize:raise RuntimeError('unsafe optimized Python')
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
def pins():
 return {n:digest(BASE/n) for n in PREFIT}
def sources(stage):
 inputs={n:BASE/n for n in SOURCES}|{n:FROZEN/n for n in ('observations.json','forge-receipt.json')}
 if stage=='verify':
  prev=next((E/'emit'/'runs').iterdir()); rec=read(prev/'run.json')
  assert verify_run(prev)['valid'] and rec['status']=='succeeded'
  for n in ('messages.json','clock.json','cache.json'):
   inputs[n]=prev/'output'/n;assert digest(inputs[n])==rec['artifacts'][n]
  inputs['private-reference.json']=FROZEN/'private-reference.json'
 return inputs
def recipe(stage, inputs):
 contract={'emit':(['messages.json','clock.json','cache.json'],'messages'),'verify':(['eligibility.json','complete-clock.json'],'groups')}
 artifacts,metric=contract[stage]
 return dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',source_kind='CONSTRUCTED',split_identity='constructed held paired groups only',name='cog065-e1-'+stage,description='Deterministic exact codec eligibility; no fit; not qualification',command=[sys.executable,'worker.py',stage],seed=65065,timeout_seconds=900,inputs=list(inputs),artifacts=artifacts,metrics={metric:'max'},budget={'cpu':1})
def main():
 if sys.argv[1:] not in (['emit'],['verify']):raise ValueError('explicit stage only')
 stage=sys.argv[1];p=BASE/f'{stage.upper()}_REVIEW.json'
 if not p.exists():raise RuntimeError('exact independent stage review missing')
 review=read(p);manifest=BASE/'PREFLIGHT_SHA256.json'
 assert read(manifest)==pins() and review['manifest_sha256']==digest(manifest) and review['decision']==f'{stage.upper()}_ONLY_GO' and review['reviewer']!='advance_csm_comparison'
 inputs=sources(stage);assert review['inputs']=={n:digest(f) for n,f in inputs.items()}
 inputs[p.name]=p;inputs[manifest.name]=manifest
 os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 ctrl=Controller(E/'ledger')
 if stage=='emit':
  ctrl.program('cog065-e1','Constructed deterministic channel feasibility, no fit',1800,0)
  put_node(ctrl,'cog065-e1.protocol','experiment',{'pins':pins(),'review':review})
 work=E/stage;work.mkdir(parents=True,exist_ok=False)
 for n,f in inputs.items():shutil.copyfile(f,work/n)
 rp=work/'recipe.json';write(rp,recipe(stage,inputs));load_config(rp)
 run=run_experiment(rp,work/'runs');rid=run['id'];v=verify_run(work/'runs'/rid)
 node='cog065-e1.'+stage+'.'+rid
 put_node(ctrl,node,'evidence',{'run':run,'verified':v});link(ctrl,node,'tests','cog065-e1.protocol')
 print(rid,run['status'],v['valid'])
 if run['status']!='succeeded' or not v['valid']:raise RuntimeError('retained failed stage; do not auto-retry')
if __name__=='__main__':main()
