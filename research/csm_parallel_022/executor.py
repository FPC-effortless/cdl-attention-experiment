"""Reconstructed exact typed execution; parallel banks are engineered, not learned.
Commands are (op,dst,src,value,expected), op0..6. None is absent; uint32
payload0 is a valid value. All dependency planning is inside the timed C call.
"""
from pathlib import Path
import ctypes,hashlib,subprocess,time,json
import numpy as np
ROOT=Path(__file__).resolve().parent
NAMES=('NOOP','SET','COPY','LINK','DELETE','QUERY','IFSET')
def build(directory=None):
 source=ROOT/'typed_executor.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest();out=Path(directory or ROOT/'build');out.mkdir(exist_ok=True,parents=True);lib=out/('executor_'+digest[:16]+'.so')
 receipt=lib.with_suffix('.json')
 if not lib.exists():
  started=time.perf_counter();command=['g++','-std=c++17','-O3','-fopenmp','-shared','-fPIC',str(source),'-o',str(lib)]
  subprocess.run(command,check=True)
  receipt.write_text(json.dumps({'source_sha256':digest,'binary_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'command':command,'compiler':subprocess.run(['g++','--version'],capture_output=True,text=True,check=True).stdout,'compile_seconds':time.perf_counter()-started},indent=2))
 if not receipt.exists():raise ValueError('missing compiled-library receipt; rebuild into fresh directory')
 r=json.loads(receipt.read_text())
 if r['source_sha256']!=digest or r['binary_sha256']!=hashlib.sha256(lib.read_bytes()).hexdigest():raise ValueError('compiled-library drift')
 return lib
class Executor:
 def __init__(self,library=None):
  self.lib=ctypes.CDLL(str(library or build()));p=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')
  self.lib.execute.argtypes=[p,p,ctypes.c_int64,p,ctypes.c_int64,p,ctypes.c_int,ctypes.c_int,ctypes.c_int,p,p];self.lib.execute.restype=ctypes.c_int
 def execute(self,operations,n=4,state=None,bank_ids=None,threads=2,parallel=True):
  started=time.perf_counter()
  if type(n)!=int or n<1 or type(threads)!=int or threads<1:raise ValueError('positive dimensions')
  ops=np.asarray(operations)
  if ops.size==0:ops=np.empty((0,5),dtype=np.int64)
  if ops.ndim!=2 or ops.shape[1]!=5 or not np.issubdtype(ops.dtype,np.integer):raise ValueError('integer[T,5] operations')
  ops=np.ascontiguousarray(ops,dtype=np.int64)
  if state is None:k=np.zeros(n,dtype=np.int64);d=k.copy()
  else:
   k,d=(np.asarray(x) for x in state)
   if any(x.shape!=(n,) or not np.issubdtype(x.dtype,np.integer) for x in (k,d)):raise ValueError('integer state vectors')
   k=np.array(k,dtype=np.int64,copy=True);d=np.array(d,dtype=np.int64,copy=True)
  banks=np.arange(n,dtype=np.int64) if bank_ids is None else np.asarray(bank_ids)
  if banks.shape!=(n,) or not np.issubdtype(banks.dtype,np.integer) or np.any(banks<0):raise ValueError('bank ids')
  # Compress arbitrary literal-derived bank IDs without changing equivalence.
  _,banks=np.unique(banks,return_inverse=True);banks=np.ascontiguousarray(banks,dtype=np.int64)
  answers=np.empty(len(ops),dtype=np.int64);stats=np.zeros(3,dtype=np.int64)
  rc=self.lib.execute(k,d,n,ops.reshape(-1),len(ops),banks,int(banks.max())+1,threads,int(parallel),answers,stats)
  if rc:raise ValueError('invalid typed program/state code '+str(rc))
  return {'state':(k.tolist(),d.tolist()),'answers':[None if x==-1 else int(x) for x in answers], 'stats':{'cross_bank_fallback':bool(stats[0]),'actual_team_threads':int(stats[1]),'banks':int(stats[2]),'full_call_seconds':time.perf_counter()-started}}
def reference(operations,n=4,state=None):
 """Independent serial Python semantics derived from pinned019 Memory contract."""
 k,d=([0]*n,[0]*n) if state is None else tuple(list(x) for x in state)
 def read(i):
  seen=set()
  while i not in seen:
   seen.add(i)
   if k[i]==0:return None
   if k[i]==1:return d[i]
   i=d[i]
  return None
 answers=[]
 for op,dst,src,v,g in operations:
  answer=-2
  if op==1:k[dst],d[dst]=1,v
  elif op==2:
   x=read(src);k[dst],d[dst]=(0,0) if x is None else (1,x)
  elif op==3:k[dst],d[dst]=2,src
  elif op==4:k[dst],d[dst]=0,0
  elif op==5:answer=read(dst)
  elif op==6:
   if read(src)==g:k[dst],d[dst]=1,v
  elif op!=0:raise ValueError(op)
  answers.append(answer)
 return {'state':(k,d),'answers':answers}
