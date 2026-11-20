import ast,hashlib,json,pathlib,subprocess,itertools,os
import torch
B=pathlib.Path(__file__).resolve().parent;S=B/'study_frozen.py';tree=ast.parse(S.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='carry');env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(S),'exec'),env);carry=env['carry'];torch.set_num_threads(1)
def oracle(initial,actions):
 c=initial.clone()
 for a in actions.unbind(1):c=torch.where(a>0,a,c)
 return c
counts={};outputs={}
for length in range(1,6):
 seq=list(itertools.product(range(5),repeat=length));actions=torch.tensor(seq).repeat(5,1);initial=torch.arange(5).repeat_interleave(len(seq));actual=carry(initial,actions);expected=oracle(initial,actions);assert torch.equal(actual,expected);counts[str(length)]=len(initial);outputs[str(length)]=hashlib.sha256(json.dumps(actual.tolist()).encode()).hexdigest()
g=torch.Generator().manual_seed(111011);randomcases=0
for length in [8,64,257,1024]:
 actions=torch.randint(0,5,(128,length),generator=g);initial=torch.randint(0,5,(128,),generator=g);actions[0]=0;actions[1]=0;actions[1,0]=1;actions[2]=0;actions[2,-1]=4
 actual=carry(initial,actions);assert torch.equal(actual,oracle(initial,actions));randomcases+=len(initial);outputs['long'+str(length)]=hashlib.sha256(json.dumps(actual.tolist()).encode()).hexdigest()
r=subprocess.run(['/tmp/lean-4.15.0-linux/bin/lean',str(B/'Carry.lean')],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
result={'scope':'latest nonzero carrier update only; forced valid actions, no learned policy proof','study_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(ast.get_source_segment(S.read_text(),node).encode()).hexdigest(),'torch':torch.__version__,'exhaustive_sequences_by_length':counts,'exhaustive_count':sum(counts.values()),'random_long_cases':randomcases,'output_hashes':outputs,'lean':{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}}
(B/'RESULTS.json').write_text(json.dumps(result,indent=2));print(result['exhaustive_count'],randomcases)
