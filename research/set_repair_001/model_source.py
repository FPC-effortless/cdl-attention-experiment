"""Load exactly hashed parent model/encoder bytes without bytecode-cache execution."""
from pathlib import Path
import hashlib,sys,types
PARENT=Path(__file__).resolve().parent.parent/'lifecycle_learning_001'
PINS={'native_csm':'f496837236ea1f53939d29776255bfba8f1d02512a8b52f581a79d94134a4234','data':'46122640ad4ead7bf1d90bf281f484fa5041afae765bb59af974f74d8126dfff','models':'7b8e85676846b68b1bb849156b755ad74fb5e32506fc0ef4bcd817f8fa5bc64a'}
for name,pin in PINS.items():
    p=PARENT/(name+'.py');source=p.read_bytes()
    if hashlib.sha256(source).hexdigest()!=pin:raise ValueError('parent source changed: '+name)
    module=types.ModuleType(name);module.__file__=str(p);sys.modules[name]=module
    exec(compile(source,str(p),'exec'),module.__dict__)
from data import encode,history_map,VOCAB,TOKENS,HEADS,CUTOFF
from models import build,tensors,objective
