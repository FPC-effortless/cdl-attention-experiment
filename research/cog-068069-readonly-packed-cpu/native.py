"""Explicit independently scoped TrainLab research stages; no default launch."""
import hashlib,json,os,sys,shutil
from pathlib import Path
from trainlab.runner import digest,load_config,run_experiment,verify_run
from trainlab.research_adequacy import recipe_identity
from research_lab.core import Controller
from research_lab.evidence import put_node,link
if sys.flags.optimize:raise RuntimeError('unsafe optimized Python')
ROOT=Path(__file__).resolve().parent
F=ROOT/'frozen'
E=ROOT/'execution'
STAGES=('fit','predict','verify')
SOURCES=('PROTOCOL.md','gateway.py','learner.py','serve.py','worker.py','execution_observer.py','test_gateway.py','test_batch.py','freeze.py','plant061.py','forage.py','native.py')
POINTS={
 'fit':('effective_rows',['models.json','fit-summary.json'],(36288,293,3)),
 'predict':('requests',['predictions.json','timing.json'],(192,48,5)),
 'verify':('groups',['summary.json'],(96,3,3)),
}
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def source_pins():return {n:digest(ROOT/n) for n in SOURCES}
def sha_inputs(d):return {n:digest(p) for n,p in d.items()}
def lineage(stage):
 p=E/stage/'runs';runs=list(p.iterdir());assert len(runs)==1
 run=runs[0];v=verify_run(run);r=read(run/'run.json')
 assert v['valid'] and r['status']=='succeeded' and r['research_adequacy']=='EXECUTED_CRITERIA_MET'
 for n,h in r['artifacts'].items():assert digest(run/'output'/n)==h
 return run,r
def closure(stage):
 src={n:ROOT/n for n in ('PROTOCOL.md','gateway.py','learner.py','serve.py','worker.py','execution_observer.py')}
 wanted={
  'fit':('train.json','train-forge-source.json','train-forge-receipt.json','cal.json','cal-forge-source.json','cal-forge-receipt.json','fit-settings.json'),
  'predict':('requests.json','requests-forge-source.json','requests-forge-receipt.json'),
  'verify':('requests.json','requests-forge-source.json','requests-forge-receipt.json','truth.json'),
 }[stage]
 src|={n:F/n for n in wanted}
 if stage=='predict':
  run,r=lineage('fit');src['models.json']=run/'output/models.json'
 if stage=='verify':
  run,r=lineage('predict')
  src|={n:run/'output'/n for n in ('predictions.json','timing.json')}
 return src
def stage_recipe(stage,folder,inputs,reviewer):
 folder.mkdir(exist_ok=False,parents=True)
 for n,p in inputs.items():shutil.copyfile(p,folder/n)
 metric,requested,(rows,size,count)=POINTS[stage]
 bounds={'intended_rows':rows,'size_limit':size,'phase_arities':count,'fixed_cpu_threads':1}
 write(folder/'bounds.json',bounds)
 files=list(inputs)+['bounds.json','adequacy.json']
 config={'schema':'trainlab.experiment.v1','execution_purpose':'research','command':[sys.executable,'worker.py',stage],'name':'cog068069-'+stage,'description':'Fixed read-only advisory experiment; scope constructed and not qualified','seed':69069,'timeout_seconds':900,'inputs':files,'artifacts':requested+['execution-observations.json'],'metrics':{metric:'max'},'source_kind':'CONSTRUCTED','split_identity': 'TRAIN+CAL only' if stage=='fit' else 'grouped fresh constructed DEV','budget':{'cpu':1}}
 dims=('model_size','training_duration','datasets','training_regime')
 plans=(('size_limit','le',size),('phase_arities','eq',count),('intended_rows','eq',rows),('fixed_cpu_threads','eq',1))
 criteria={}
 for d,(f,op,h) in zip(dims,plans):
  criteria[d]={'configuration':'Fixed bounded '+stage+' '+d,'justification':'Required objective-specific pilot design condition','limitations':'Necessary condition only, no convergence, no qualification','criteria':[{'id':d,'rationale':'Prospectively declared bounded check','evidence':'bounds.json','field':f,'operator':op,'threshold':h}]}
 measured=(('size','le',size),('wall_seconds','le',900),('rows','eq',rows),('stages','eq',count))
 gate={'schema':'trainlab.research-adequacy.v1','claim':'Bounded read-only advice and packed CPU parity comparison','claim_kind':'bounded_diagnostic','scope':'Constructed local physical monitor only','excluded_claims':'Authentic interactive, hardware, sparsity, actuator, finalist or foundation claims','recipe_sha256':recipe_identity(config),'inputs':{n:digest(folder/n) for n in files if n!='adequacy.json'},'review':{'author':'advance_csm_comparison','reviewer':reviewer,'basis':'Independent stage review and runtime/source closure','criteria_frozen_before_evaluation':True},'dimensions':criteria,'execution':{'artifact':'execution-observations.json','dimensions':{d:[{'field':f,'operator':op,'threshold':h,'rationale':'Post-run model/work/data/regime claim bound'}] for d,(f,op,h) in zip(dims,measured)}}}
 gate['execution']['dimensions']['training_regime'].append({'field':'blas_threads','operator':'eq','threshold':1,'rationale':'Observed local CPU only'})
 write(folder/'adequacy.json',gate);config['research_adequacy']={'file':'adequacy.json','sha256':digest(folder/'adequacy.json')}
 p=folder/'recipe.json';write(p,config);load_config(p)
 return p
def main():
 if sys.argv[1:] not in ([x] for x in STAGES):raise RuntimeError('explicit stage only')
 stage=sys.argv[1];p=ROOT/(stage.upper()+'_REVIEW.json');srcpin=ROOT/'SOURCE_FREEZE.json'
 if not p.exists():raise RuntimeError('requires exact independent stage review')
 r=read(p);assert r['decision']==stage.upper()+'_ONLY_GO' and r['reviewer']!='advance_csm_comparison' and r['source_freeze_sha256']==digest(srcpin)
 assert read(srcpin)==source_pins()
 src=closure(stage);assert r['inputs']==sha_inputs(src)
 # Native runner does not load review code. Require independently validated live
 # runtime pins in caller environment, and retain them as a scientific input.
 runtime=ROOT/'APPROVED_RUNTIME.json';assert r['runtime_sha256']==digest(runtime) and os.environ['COG_068069_VERIFIED_RUNTIME_SHA256']==digest(runtime)
 src|={'review.json':p,'runtime.json':runtime,'source-freeze.json':srcpin}
 os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 c=Controller(E/'ledger')
 if stage=='fit':
  c.program('cog068069','Constructed read-only advice and CPU parity',2700,0)
  put_node(c,'cog068069.protocol','experiment',{'sources':source_pins(),'first_review':r})
 folder=E/stage;recipe=stage_recipe(stage,folder,src,r['reviewer']);result=run_experiment(recipe,folder/'runs');v=verify_run(folder/'runs'/result['id'])
 node='cog068069.'+stage+'.'+result['id'];put_node(c,node,'evidence',{'native_run':result,'verified':v});link(c,node,'tests','cog068069.protocol')
 print(result['id'],result['status'],result['research_adequacy'],v['valid'])
 if result['status']!='succeeded' or not v['valid']:raise RuntimeError('failed run retained; no retries')
if __name__=='__main__':main()
