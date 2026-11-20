"""Assemble a pinned private code/asset closure without importing it or changing sources."""
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def safe_relative(name):
 q=PurePosixPath(name)
 if q.is_absolute() or '..' in q.parts or not q.parts:raise ValueError('Unsafe manifest path')
 return q

def assemble(code_root,asset_root,destination,expected_inputs_sha):
 code_root=Path(code_root);asset_root=Path(asset_root);destination=Path(destination)
 if destination.exists():raise ValueError('Destination already exists')
 if digest(code_root/'INPUTS.json')!=expected_inputs_sha:raise ValueError('Input manifest mismatch')
 entries=json.loads((code_root/'INPUTS.json').read_text())['files']
 # Administrative records are checked against admission inside native adapter.
 for name in ('INPUTS.json','REVIEW.json','ADMISSION.json'):
  if name not in entries:entries[name]=digest(code_root/name)
 sources={}
 for name,sha in entries.items():
  q=safe_relative(name);candidate=code_root.joinpath(*q.parts)
  if candidate.is_file():choices=[candidate]
  else:choices=list(asset_root.rglob(q.name))
  matches=[]
  for p in choices:
   if not p.is_file() or any(x.is_symlink() for x in (p,*p.parents)):continue
   if digest(p)==sha:matches.append(p)
  if not matches:raise ValueError('Missing exact input: '+name)
  sources[name]=matches[0]
 destination.mkdir(mode=0o700,parents=True)
 for name,p in sources.items():
  target=destination.joinpath(*safe_relative(name).parts)
  target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
  for parent in [target.parent,*target.parent.parents]:
   if parent==destination.parent:break
   os.chmod(parent,0o700)
  with open(p,'rb') as src,open(target,'xb') as dst:shutil.copyfileobj(src,dst)
  os.chmod(target,0o600)
  if digest(target)!=entries[name]:raise ValueError('Copy verification failed: '+name)
 return {'files':len(sources),'inputs_sha256':expected_inputs_sha,'private_modes':'0700 directories,0600 files'}
