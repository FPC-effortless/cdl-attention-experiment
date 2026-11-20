"""Verify fresh-fit checkpoint predictions against the recorded public cohort."""
import hashlib,json,os,shutil
from pathlib import Path
import joblib
from authentic import banking,issues
from language import Grounder
ROOT=Path(__file__).resolve().parent

def main():
    store=Path(os.environ['COG_CHECKPOINT_DIR'])
    for name,fit in [('banking',banking),('issues',issues)]:
        fresh=fit();old=json.loads((ROOT/'results'/('authentic-'+name+'.json')).read_bytes())
        assert fresh['source']==old['source']
        for arm,x in fresh['arms'].items():assert x['rows']==old['arms'][arm]['rows']
    public=ROOT/'checkpoints';public.mkdir(exist_ok=True)
    for arm,expanded in [('word',False),('expanded',True)]:
        joblib.dump(Grounder(expanded),public/('issue-simulator-'+arm+'.joblib'),compress=3)
        shutil.copyfile(store/'banking'/(arm+'.joblib'),public/('banking-'+arm+'.joblib'))
    digests={'restricted-github-'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (store/'issues').iterdir()}
    (ROOT/'results/checkpoints.json').write_text(json.dumps(digests,indent=2))
if __name__=='__main__':main()
