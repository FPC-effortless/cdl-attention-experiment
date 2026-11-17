"""Runtime identity check, not hardware attestation or package supply-chain proof."""
import sys,platform,hashlib,json,importlib.metadata
from pathlib import Path
import torch
def fingerprint():
 return dict(python=sys.version,platform=platform.platform(),torch_version=torch.__version__,torch_git=torch.version.git_version,torch_config=torch.__config__.show(),packages=sorted([d.metadata['Name'],d.version] for d in importlib.metadata.distributions()),torch_init_sha256=hashlib.sha256(Path(torch.__file__).read_bytes()).hexdigest())
def verify(root):
 expected=json.loads((Path(root)/'RUNTIME_LOCK.json').read_text())
 if fingerprint()!=expected:raise RuntimeError('runtime fingerprint drift; quarantine until reviewed')
 return expected
if __name__=='__main__':Path('RUNTIME_LOCK.json').write_text(json.dumps(fingerprint(),sort_keys=True,indent=2))
