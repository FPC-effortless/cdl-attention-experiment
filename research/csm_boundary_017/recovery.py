"""Local immutable generations + atomic index. No host-durability attestation."""
import os,hashlib,json,random,torch,uuid
from pathlib import Path

def _sync_file(p):
 with p.open('rb') as f:os.fsync(f.fileno())
def save(path,model,optimizer,sampler,update,identity):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);gens=path.parent/(path.name+'.generations');gens.mkdir(exist_ok=True)
 payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler':sampler.getstate(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'completed_updates':update,'identity':identity}
 gen=uuid.uuid4().hex;target=gens/(gen+'.pt');torch.save(payload,target);_sync_file(target)
 receipt={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'completed_updates':update,'identity':identity,'generation':gen}
 rp=gens/(gen+'.json');rp.write_text(json.dumps(receipt,sort_keys=True));_sync_file(rp)
 tmp=path.with_name(path.name+'.'+gen+'.tmp');tmp.write_text(json.dumps(receipt,sort_keys=True));_sync_file(tmp);os.replace(tmp,path)
 return receipt

def restore(path,model,optimizer,sampler,identity):
 path=Path(path);r=json.loads(path.read_text());gen=r['generation']
 if not isinstance(gen,str) or len(gen)!=32 or any(c not in '0123456789abcdef' for c in gen):raise ValueError('invalid generation')
 target=path.parent/(path.name+'.generations')/(gen+'.pt');rp=target.with_suffix('.json')
 if json.loads(rp.read_text())!=r or r['identity']!=identity or hashlib.sha256(target.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('checkpoint identity/hash mismatch')
 p=torch.load(target,weights_only=True)
 if p['identity']!=identity or p['completed_updates']!=r['completed_updates']:raise ValueError('checkpoint receipt mismatch')
 model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer']);sampler.setstate(p['sampler']);torch.set_rng_state(p['torch_rng']);random.setstate(p['python_rng']);return p['completed_updates']
