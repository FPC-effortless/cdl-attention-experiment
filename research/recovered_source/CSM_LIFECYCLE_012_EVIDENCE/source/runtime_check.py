import sys,json,hashlib,importlib
from pathlib import Path
import torch
R=Path(__file__).parent
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def runtime_check():
 spec=json.loads((R/'runtime.json').read_text());require(sys.version==spec['python'] and sys.executable==spec['python_executable'] and sha(sys.executable)==spec['python_executable_sha256'],'Python drift');require(torch.__version__==spec['torch_version'] and torch.__config__.show()==spec['torch_config'],'Torch drift')
 for n,d in spec['files'].items():
  m=importlib.import_module(n);require(str(Path(m.__file__))==d['path'] and sha(m.__file__)==d['sha256'],'module drift')
