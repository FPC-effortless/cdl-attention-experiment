"""Bind declared runtime observations to exact native execution input closure."""
import hashlib,json,os
from threadpoolctl import threadpool_info
from pathlib import Path
from trainlab.research_adequacy import recipe_identity
def write(observations):
 pools=threadpool_info();assert pools and all(x['num_threads']==1 for x in pools)
 observations=dict(observations,blas_threads=max(x['num_threads'] for x in pools))
 p=Path('recipe.json');recipe=json.loads(p.read_text());inputs={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in recipe['inputs']}
 receipt={'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(recipe),'inputs':inputs,'producer':{'file':'execution_observer.py','sha256':inputs['execution_observer.py']},'observations':observations}
 (Path(os.environ['TRAINLAB_OUTPUT_DIR'])/'execution-observations.json').write_text(json.dumps(receipt,allow_nan=False))
