"""Private OUTPUT-MEMORY-021 bounded one-seed scientific study; no protected TEST or paid compute."""
import os,sys,subprocess,hashlib,json,traceback,zipfile,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
INPUTS_SHA='4a937291f70ba196b67a7a0a0dd2ee53ccff3f1117fce24ac635c3483b0a209e'
ASSEMBLER_SHA='0afb5173b61e8f5ad06130bad4abce563c25e3b260ae853687a02ff70fbcdae4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 roots=[p.parent for p in Path('/kaggle/input').rglob('INPUTS.json') if sha(p)==INPUTS_SHA]
 assert len(roots)==1,'Expected unique admitted code closure'
 root=roots[0];assembler=root/'assemble_private_inputs.py'
 assert sha(assembler)==ASSEMBLER_SHA,'Transport source mismatch'
 # Source executes only after launcher-pinned hash verification.
 ns={'__name__':'admitted_input_assembler'};exec(compile(assembler.read_text(),str(assembler),'exec'),ns)
 receipt=ns['assemble'](root,Path('/kaggle/input'),Path('/tmp/woyengi-output021-scientific'),INPUTS_SHA)
 Path('/kaggle/working/ASSEMBLY.json').write_text(json.dumps(receipt,indent=2))
 try:
  subprocess.run([sys.executable,'/tmp/woyengi-output021-scientific/native_scientific021.py','run','/tmp/woyengi-output021-scientific','/tmp/output021_scientific_result'],check=True)
 finally:
  archive=Path('/tmp/output021_scientific_result.zip')
  if archive.exists():
   with zipfile.ZipFile(archive) as z:assert z.testzip() is None,'Output ZIP integrity failed'
   exported=Path('/kaggle/working/output021_scientific_result.zip');shutil.copyfile(archive,exported)
   assert sha(exported)==sha(archive),'Output archive copy mismatch'
except BaseException as e:
 Path('/kaggle/working/LAUNCH_FAILURE.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc(),'scope':'admitted one-seed study only; no protected TEST or paid compute'},indent=2));raise
