"""Pinned native Forge → TrainLab → study scoring adapter → Research Lab runner."""
import json,os,sys,shutil,hashlib,platform,importlib.metadata
from pathlib import Path
from trainlab.runner import run_experiment, verify_run, digest
from trainlab.research_adequacy import recipe_identity
from research_lab.core import Controller
from research_lab.evidence import put_node, link
if sys.flags.optimize:raise RuntimeError('unsafe optimized execution')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'execution'
SOURCE=ROOT/'frozen'
PINS=('PROTOCOL.md','sim.py','model.py','fit.py','predict.py','score_veritas.py','lab.py','test_preflight.py','observe.py','RUNTIME.json','audit_generator.py','DATA_AUDIT.json')
DATA=('train.json','challenge.json','truth.json','counts.json','train-forge.json','challenge-forge.json')
def snap():return {str(x.relative_to(ROOT)):digest(x) for x in [*(ROOT/p for p in PINS),*(SOURCE/d for d in DATA)]}
def runtime():
 import numpy, threadpoolctl
 import trainlab.runner, trainlab.execution_adequacy, trainlab.research_adequacy, forge_data.diagnostics, research_lab.core, research_lab.evidence
 m=(trainlab.runner,trainlab.execution_adequacy,trainlab.research_adequacy,forge_data.diagnostics,research_lab.core,research_lab.evidence)
 return {'ecosystem_commit':'eec439492407b0aa26922c23b5ae40d95289f4eb','python':platform.python_version(),'executable_sha256':digest(Path(sys.executable).resolve()),'numpy':importlib.metadata.version('numpy'),'threadpoolctl':importlib.metadata.version('threadpoolctl'),'blas':[{'version':v['version'],'implementation':v['internal_api'],'sha256':digest(Path(v['filepath']))} for v in threadpoolctl.threadpool_info()],'module_sha256':{x.__name__:digest(Path(x.__file__)) for x in m}}
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def review():
 pin=ROOT/'PREFIT_SHA256.json';report=ROOT/'INDEPENDENT_REVIEW.json'
 if not pin.exists() or not report.exists():raise RuntimeError('PREFIT_REVIEW_MISSING; refusing all fit/predict/score')
 p=load(pin);r=load(report)
 assert load(ROOT/'RUNTIME.json')==runtime()
 assert p==snap() and digest(pin)==r['manifest_sha256'] and r['decision']=='GO' and r['reviewer']!='advance_csm_comparison' and r['reviewer']
 return p,r
def stage(label,cmd,sources,artifacts,metric,folder,reviewer,review_file):
 folder.mkdir(parents=True,exist_ok=False)
 inputs=dict(sources);inputs['observe.py']=ROOT/'observe.py';inputs['counts.json']=SOURCE/'counts.json';inputs['PROTOCOL.md']=ROOT/'PROTOCOL.md';inputs['review.json']=review_file;inputs['RUNTIME.json']=ROOT/'RUNTIME.json';inputs['DATA_AUDIT.json']=ROOT/'DATA_AUDIT.json'
 for k,p in inputs.items():shutil.copyfile(p,folder/k)
 recipe={'schema':'trainlab.experiment.v1','execution_purpose':'research','name':'cog061063-'+label,'description':'Constructed bounded causal dynamics; no scientific qualification','command':[sys.executable,cmd],'seed':61063,'timeout_seconds':900,'inputs':list(inputs)+['adequacy.json'],'artifacts':artifacts+['execution-observations.json'],'metrics':{metric:'max'},'source_kind':'CONSTRUCTED','split_identity': 'constructed-train' if label=='fit' else 'constructed-grouped-DEV-only','budget':{'cpu':1}}
 rules={'model_size':[('rank','eq',4),('sensor_dim','eq',16),('lag','eq',3)],'training_duration':[('train_transitions','eq',8192),('ridge_fits_per_arm','eq',1)],'datasets':[('train_groups','ge',64),('dev_groups','ge',64),('group_intersection','eq',0),('paired_controls','eq',4)],'training_regime':[('planned_threads','eq',1),('max_seconds','le',900)]}
 dimensions={}
 for name,vs in rules.items():
  dimensions[name]={'configuration':name+' bound as frozen in PROTOCOL.md','justification':'Necessary prefit bound for this controlled fixed-budget diagnostic','limitations':'Counts describe design, not task convergence, identifiability, or generalization','criteria':[{'id':f,'rationale':'Frozen necessary condition','evidence':'counts.json','field':f,'operator':op,'threshold':v} for f,op,v in vs]}
 gate={'schema':'trainlab.research-adequacy.v1','claim':'Bounded predictive state and paired effect diagnostic','claim_kind':'bounded_diagnostic','scope':'Constructed development panel, shared known controls, fixed finite coverage','excluded_claims':'Authentic causality, discovered named concepts, frontier, qualification, capacity ceiling','recipe_sha256':recipe_identity(recipe),'inputs':{k:digest(folder/k) for k in inputs},'review':{'author':'advance_csm_comparison','reviewer':reviewer,'basis':'Exact prefit source/data manifest; independent review.json','criteria_frozen_before_evaluation':True},'dimensions':dimensions}
 expected={'fit':3,'predict':15,'verify':24}[label]
 runtime_bounds={'model_size':('model_size','le',20000),'training_duration':('work_seconds','le',900),'datasets':('groups','eq',64),'training_regime':('complete','eq',expected)}
 gate['execution']={'artifact':'execution-observations.json','dimensions':{k:[{'field':field,'operator':op,'threshold':value,'rationale':'Frozen observed count/clock/size bound for stage'}] for k,(field,op,value) in runtime_bounds.items()}}
 gate['execution']['dimensions']['training_regime'].append({'field':'blas_threads','operator':'eq','threshold':1,'rationale':'Observed BLAS threading must match the fixed CPU plan'})
 save(folder/'adequacy.json',gate);recipe['research_adequacy']={'file':'adequacy.json','sha256':digest(folder/'adequacy.json')}
 rp=folder/'recipe.json';save(rp,recipe)
 return rp
def run(label,cmd,sources,artifacts,metric,ctrl,reviewer):
 folder=OUT/label
 rp=stage(label,cmd,sources,artifacts,metric,folder,reviewer,ROOT/'INDEPENDENT_REVIEW.json')
 got=run_experiment(rp,folder/'runs');p=folder/'runs'/got['id'];v=verify_run(p)
 put_node(ctrl,'cog061063.'+label+'.'+got['id'],'evidence',{'run':got,'verification':v});link(ctrl,'cog061063.'+label+'.'+got['id'],'tests','cog061063.protocol')
 if got['status']!='succeeded' or not v['valid']:raise RuntimeError(f'Run preserved, stage failed: {got["id"]}')
 return p/'output'
def main():
 pin,r=review()
 os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 ctrl=Controller(OUT/'ledger');ctrl.program('cog061063','Bounded continuous controlled dynamics',2700,0);put_node(ctrl,'cog061063.protocol','experiment',{'files':pin,'review':r})
 fit=run('fit','fit.py',{'fit.py':ROOT/'fit.py','model.py':ROOT/'model.py','train.json':SOURCE/'train.json','train-forge.json':SOURCE/'train-forge.json'},['predictive.npz','pca.npz','narx.npz','fit-summary.json'],'fits',ctrl,r['reviewer'])
 models={k+'.npz':fit/(k+'.npz') for k in ('predictive','pca','narx')}
 pred=run('predict','predict.py',{'predict.py':ROOT/'predict.py','model.py':ROOT/'model.py','challenge.json':SOURCE/'challenge.json','challenge-forge.json':SOURCE/'challenge-forge.json'}|models,['predictions.json','timing.json','held-encodings.json'],'requests_per_arm',ctrl,r['reviewer'])
 run('verify','score_veritas.py',{'score_veritas.py':ROOT/'score_veritas.py','model.py':ROOT/'model.py','truth.json':SOURCE/'truth.json','train.json':SOURCE/'train.json','predictions.json':pred/'predictions.json','held-encodings.json':pred/'held-encodings.json'}|models,['scores.json','group-effects.json'],'groups',ctrl,r['reviewer'])
if __name__=='__main__':main()
