import time
from observe import write
import json,os,sys,time
from pathlib import Path
import numpy as np
from model import fit_arm,read_view
if sys.flags.optimize:raise RuntimeError('unsafe optimized execution')
def main():
 started=time.perf_counter()
 inp=Path('train.json')
 assert not Path('truth.json').exists() and not Path('challenge.json').exists()
 data=read_view(inp)
 assert len(data)==64 and all(o.shape==(129,16) and u.shape==(128,2) for o,u in data.values())
 out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);v={}
 for arm in ('predictive','pca','narx'):
  start=time.perf_counter(); cpu_start=time.process_time();m=fit_arm(data,arm);elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu_start
  assert all(np.all(np.isfinite(x)) for x in m.values())
  np.savez(out/(arm+'.npz'),**m)
  v[arm]={'fit_seconds':elapsed,'fit_cpu_seconds':cpu,'usable_regression_rows':sum(len(u)-2 for _,u in data.values()),'stored_scalars':sum(np.size(x) for k,x in m.items() if k!='gram_cond'),'gram_condition':float(m['gram_cond'])}
 (out/'fit-summary.json').write_text(json.dumps(v,allow_nan=False))
 (out/'metrics.json').write_text(json.dumps({'fits':len(v)}))
 write({"complete":len(v),"groups":len(data),"model_size":max(x["stored_scalars"] for x in v.values()),"work_seconds":sum(x["fit_seconds"] for x in v.values())})
if __name__=='__main__':main()
