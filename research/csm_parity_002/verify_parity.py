"""Hard-forward archived PyTorch/reference parity; no training claims."""
import hashlib, importlib.util, json, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/csm_regulation_math_001'))
import reference as ref
archive=ROOT/'artifacts/csm-original-scope-reconstructed-20261109/CSM-GRAPH-001-verified-project.zip'
member='CSM-GRAPH-001/csm-foundation/experiments/csm_foundation_001/model.py'
source=zipfile.ZipFile(archive).read(member)
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'archived.py';path.write_bytes(source)
    spec=importlib.util.spec_from_file_location('archived',path)
    mod=importlib.util.module_from_spec(spec);sys.modules['archived']=mod;spec.loader.exec_module(mod)

mapping={'embedding':'embedding.weight','operation_embedding':'operator_embedding','first':'first_address','second':'second_address','gain':'write_gate','transform1':'transform.0','transform2':'transform.2'}
rows=[]
torch.set_num_threads(1)
for dtype in (torch.float64,torch.float32):
 for seed in range(6):
  for mode in ('rewrite','ring_write','no_links'):
   c=ref.Config(traces=0); p=ref.parameters(c,seed)
   model=mod.RecordRewrite(mod.Config(vocab_size=c.vocabulary,mode=mode)).to(dtype=dtype)
   weights={}
   for key,value in p.items():
    if key in ('embedding','operation_embedding'): target=mapping[key]
    else:
     stem,suffix=key.rsplit('.',1);target=mapping.get(stem,stem)+('.weight' if suffix=='W' else '.bias')
    weights[target]=torch.tensor(value,dtype=dtype)
   model.load_state_dict(weights)
   st=model.init_state(1,'cpu');nr=ref.initialize(c)
   tokens=np.random.default_rng(1000+seed).integers(0,c.vocabulary,96)
   maximum=0.; decisions=0
   with torch.no_grad():
    for token in tokens:
     nr,y,tr=ref.step(c,p,nr,int(token),ref.decay(c,'instant'),mode)
     ty,st,tt=model.step(torch.tensor([token]),st,return_trace=True)
     for k in ('first','second','destination','operator'):
      assert tr[k]==tt[k].item(),(dtype,seed,mode,k);decisions+=1
     assert tr['admitted']==bool(tt['edits_value'].item())
     assert tr['link_edit']==bool(tt['edits_link'].item())
     assert nr.cursor==st.cursor.argmax(-1).item()
     np.testing.assert_array_equal(nr.links,st.links[0].argmax(-1).numpy())
     assert nr.position==st.position
     for a,b in ((y,ty[0].numpy()),(nr.values,st.values[0].numpy()),(np.stack((nr.older,nr.recent)),st.history[0].numpy())):
      maximum=max(maximum,float(np.max(np.abs(a-b))))
      np.testing.assert_allclose(a,b,rtol=2e-5 if dtype==torch.float32 else 1e-11,atol=2e-6 if dtype==torch.float32 else 1e-12)
   rows.append(dict(dtype=str(dtype),seed=seed,mode=mode,steps=len(tokens),route_checks=decisions,max_abs_error=maximum))
result=dict(status='PASS',torch=torch.__version__,numpy=np.__version__,archived_source_sha256=hashlib.sha256(source).hexdigest(),rows=rows,scope='Baseline hard forward only; no regulated-runtime or gradient-estimator parity; no learning claim')
print(json.dumps(result,indent=2))
