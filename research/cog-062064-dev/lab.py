import os,sys,json,hashlib,shutil
from pathlib import Path
if sys.flags.optimize: raise RuntimeError('Optimized Python disables research integrity checks')
from forge_data.core import canonical
from forge_data.diagnostics import diagnose
from trainlab.runner import run_experiment,verify_run
from research_lab.core import Controller
from research_lab.evidence import put_node,link
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'cog-attn-handoff/continuation/cog-051-060';DATA=HERE/'data';LAB=HERE/'lab'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
 pins=json.loads((DATA/'pins.json').read_text());rows=[]
 for file,h in pins.items():
  assert file.startswith('dev/') and digest(DATA/file)==h
  for d in json.loads((DATA/file).read_text()):
   history=[];old=['<empty>']*2
   for t in d['turns']:
    text=t['utterance'].lower()
    if t['speaker']=='USER':
     labels={k:v for f in t['frames'] for k,v in f['state']['slot_values'].items()};truth=[labels.get(k,['<empty>'])[0] for k in ('hotel-area','hotel-pricerange')];delta=[v if v!=o else '<hold>' for v,o in zip(truth,old)]
     row={'id':d['dialogue_id'],'turn':int(t['turn_id']),'text':[text,' '.join(history[-2:])],'truth':truth,'prior':old,'delta':delta}
     rows.append({'id':d['dialogue_id']+':'+str(t['turn_id']),'split':'validation','features':{'row':row,'text':row['text']},'groups':{'dialogue':d['dialogue_id']},'label':delta});old=truth
    history.append(text)
 view=HERE/'forge-view.json';data=canonical({'schema':'forge.diagnostic-input.v1','feature_fields':['row','text'],'group_fields':['dialogue'],'rows':rows})
 if view.exists():assert view.read_bytes()==data
 else:view.write_bytes(data)
 report=diagnose(view,sha256=digest(view),max_rows=10000);(HERE/'forge-receipt.json').write_text(json.dumps(report,indent=2));return view

def main():
 view=freeze();LAB.mkdir(exist_ok=True);controller=Controller(LAB/'os');controller.program('cog062064','DEV uncertainty and persistent-error diagnostic',1800,0)
 files=['study.py','runtime.py','PROTOCOL.md','forge-view.json','forge-receipt.json','upstream/belief.py','upstream/state_store.py']
 pin=put_node(controller,'cog062064.protocol','experiment',{x:digest(HERE/x) for x in files})
 os.environ.update(COG_BASE=str(BASE),COG_DATA=str(DATA),FORGE_VIEW=str(view),FORGE_SHA256=digest(view),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 mode=sys.argv[1] if len(sys.argv)>1 else 'score';folder=LAB/mode;folder.mkdir(exist_ok=True)
 for name in files:
  dest=folder/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(HERE/name,dest)
 artifacts=['timing.json','recovery_timing.json'] if mode=='timing' else ['results.json','traces.json','metrics_detail.json']
 recipe={'schema':'trainlab.experiment.v1','execution_purpose':'engineering_diagnostic','name':'cog-062064-'+mode,'description':'Previously exposed authentic DEV; not qualification or training','command':[sys.executable,'study.py']+(['timing'] if mode=='timing' else []),'seed':62064,'timeout_seconds':1800,'inputs':files,'artifacts':artifacts,'metrics':{'timed_sessions':'max'} if mode=='timing' else {'turns':'max'},'source_kind':'AUTHENTIC_CANDIDATE','split_identity':'MultiWOZ22_OFFICIAL_DEV_ONLY_EXPOSED','budget':{'cpu':1}}
 rp=folder/'recipe.json';rp.write_text(json.dumps(recipe));record=run_experiment(rp,folder/'runs');run=folder/'runs'/record['id'];verdict=verify_run(run);receipt={'record':record,'verification':verdict,'protocol':pin,'forge_input_sha256':digest(view),'ecosystem':'eec439492407b0aa26922c23b5ae40d95289f4eb'};(HERE/(mode+'-receipt.json')).write_text(json.dumps(receipt,indent=2));assert record['status']=='succeeded' and verdict['valid'],record
 put_node(controller,'cog062064.'+mode,'evidence',receipt);link(controller,'cog062064.'+mode,'tests','cog062064.protocol')
 for name in artifacts:shutil.copyfile(run/'output'/name,HERE/name)
 print(json.dumps({'run':record['id'],'status':record['status'],'seconds':record['elapsed_seconds']}))
if __name__=='__main__':main()
