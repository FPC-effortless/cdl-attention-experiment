"""Prespecified predictions; no gold, model selection or fitting."""
import json,os
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from model import base_features,shortlist,meta_text,meta_features

def main():
    rows=json.loads(Path('E-input.json').read_bytes())
    if any(set(r)!={'dialogue','group','turn','text'} for r in rows):
        raise ValueError('prediction receives only lexical and row-alignment data')
    model=joblib.load('heads.joblib')  # locally produced and TrainLab-verified by reviewed launcher
    out=[{k:r[k] for k in ('dialogue','group','turn')}|{'slots':[]} for r in rows]
    with threadpool_limits(limits=1):
        x,_=base_features(rows,model['vectorizers'])
        text={k:meta_text(rows,k) for k in ('ordered','canonical')}
        for j,b in enumerate(model['base']):
            p=b.predict_proba(x); ids,ps=shortlist(p)
            blocks={k:meta_features(v,ids,ps) for k,v in text.items()}
            arms={}
            for tag,head,history in [('canonical','canonical','canonical'),('ordered','ordered','ordered'),('order_removed','ordered','canonical')]:
                chooser=model[head][j]
                if chooser.classes_.tolist()!=[0,1]:raise ValueError("non-binary chooser checkpoint")
                confidence=chooser.predict_proba(blocks[history])[:,1]
                if not np.isfinite(confidence).all():raise ValueError("nonfinite chooser scores")
                which=(confidence>.5).astype(int)  # tie and no positive rank2 decision stay rank1
                arms[tag]=b.classes_[ids[np.arange(len(rows)),which]].tolist()
            for i,r in enumerate(out):
                r['slots'].append({'top2':b.classes_[ids[i]].tolist()}|{tag:v[i] for tag,v in arms.items()})
    d=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    (d/'predictions.json').write_text(json.dumps(out,sort_keys=True,allow_nan=False))
    (d/'metrics.json').write_text(json.dumps({'user_turns':len(out)}))
if __name__=='__main__':main()
