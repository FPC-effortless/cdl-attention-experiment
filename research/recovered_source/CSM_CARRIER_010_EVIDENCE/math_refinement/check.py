import ast,hashlib,json,sys,pathlib,subprocess,os
import torch
from torch import nn
B=pathlib.Path(__file__).resolve().parent;S=B/'study_frozen.py';T=B.parent/'csm-math-008';from executor_frozen import Memory
text=S.read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Carrier');env={'torch':torch,'nn':nn,'CODES':torch.tensor([0,97,98,99,100])};exec(compile(ast.Module(body=[node],type_ignores=[]),str(S),'exec'),env);Carrier=env['Carrier']
class Forced(nn.Module):
 def __init__(self,a):super().__init__();self.a=a
 def forward(self,x):
  out=torch.zeros(len(x),5);out[:,self.a]=1;return out
rows=[]
with torch.no_grad():
 for conditioned in [False,True]:
  m=Carrier(conditioned)
  for code in range(5):
   for action in range(5):
    m.head=Forced(action);st=m.init(1);st=(torch.tensor([code]),st[1]);events,new=m.step(torch.tensor([97]),st);actual=int(new[0][0]);mem=Memory(1)
    if code:mem.apply('SET',0,v=code-1)
    if action:mem.apply('SET',0,v=action-1)
    expected=0 if mem.read(0) is None else mem.read(0)+1
    assert actual==expected
    rows.append({'conditioned':conditioned,'code':code,'action':action,'actual':actual,'expected':expected})
env=os.environ.copy();env['LEAN_PATH']=str(B);base=subprocess.run(['/tmp/lean-4.15.0-linux/bin/lean','-o',str(B/'TypedMemory.olean'),str(B/'TypedMemory.lean')],capture_output=True,text=True,env=env);assert base.returncode==0; r=subprocess.run(['/tmp/lean-4.15.0-linux/bin/lean',str(B/'Carrier.lean')],capture_output=True,text=True,env=env);assert r.returncode==0,r.stdout+r.stderr
out={'scope':'exact forced-action Carrier.step refinement only; no learned selection/localfrontend guarantee; no training','source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'torch':torch.__version__,'cases':rows,'lean':{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}}
(B/'RESULTS.json').write_text(json.dumps(out,indent=2));print(len(rows),'checks passed')
