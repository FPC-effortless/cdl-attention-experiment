"""COG-WEIGHT-019: Frozen-activation recoverability; posthoc diagnostic.

Loads trained 016 rankers (and optionally 018B-gate edits) and holds them frozen.
Fits diagnostic probes using 017 training labels only, chooses regularization from
017 dev only, scores both retained and novel held-out test panels. No model edits.
Only synthetic labels are used. This is NOT a production verifier.
"""
from __future__ import annotations
import sys,json,hashlib,time,csv,os
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent;P=Path(os.environ.get('COG_WEIGHT_ROOT', str(HERE.parent)))
sys.path.insert(0,str(P/'COG_WEIGHT_017_Relationship_Verification_20261122'))
import experiment_017 as r
r.b.MAXLEN=36
SEEDS=(1701,1707,1711)
torch.set_num_threads(1)

def feats(model,rows):
    """Existing 9 cosine similarities + raw head logit per document, 20 floats/row."""
    F,S=r.extract(model,rows)
    return F.detach().numpy().reshape(len(rows),20).astype('float64'),S.detach().numpy().astype('float64')

def single_candidate(feat,label):
    """Predict candidate-level support using only local features, no gold roles."""
    x=feat.reshape(len(label),2,10).reshape(-1,10)
    y=(np.repeat(np.asarray(label),2)==np.tile([0,1],len(label))).astype(int)
    return x,y

def check(model,rows):
    x,s=feats(model,rows);gold=np.asarray([z['label'] for z in rows]);known=gold>=0
    best=np.argmax(s,axis=1)
    return {'X':x,'Y':gold,'scores':s,'n':len(rows),'known':int(known.sum()),
       'absent':int((~known).sum()),'ranking':int(np.sum(known&(best==gold)))}

def binary_probe(pack,C):
    X,Y=single_candidate(pack['X'],pack['Y'])
    scale=StandardScaler().fit(X)
    clf=LogisticRegression(C=C,max_iter=300,random_state=0,solver='lbfgs')
    clf.fit(scale.transform(X),Y)
    return scale,clf

def metrics(scale,clf,pack):
    xx=pack['X'].reshape(-1,2,10).reshape(-1,10)
    p=clf.predict_proba(scale.transform(xx))[:,1].reshape(-1,2)
    y=pack['Y'];ans=np.maximum(p[:,0],p[:,1]);present=(y>=0)
    present_auc=roc_auc_score(present,ans) if present.any() and (~present).any() else None
    ys=(np.repeat(y,2)==np.tile([0,1],len(y))).astype(int)
    cand_auc=roc_auc_score(ys,p.reshape(-1)) if ys.any() and (1-ys).any() else None
    vals={'rows':int(len(y)),'known':int(present.sum()),'absent':int((~present).sum()),
         'candidate_auc':float(cand_auc) if cand_auc is not None else None,
         'presence_auc':float(present_auc) if present_auc is not None else None,
         'rank_top1':int(np.sum(present & (p.argmax(1)==y))),
         'raw_rank_top1':int(pack['ranking'])}
    return vals

def run():
  begin=time.monotonic();raw=json.loads((P/'COG_WEIGHT_017_Relationship_Verification_20261122'/'LOCKED_DATASETS.json').read_text())
  groupnames=['old_matched_test','old_history_test','old_roles_test','old_unseen_test','old_paraphrase_test','unseen_fresh','history_ood_fresh','history_long_fresh']
  groups={k:raw['groups'][k] for k in groupnames}
  # 018 completely held-out panel used ONLY for diagnostics, no selection.
  new=json.loads((P/'COG_WEIGHT_018_Circuit_Repair_20261122'/'LOCKED_COHORTS_018B.json').read_text())
  groups.update({k:new[k] for k in ['new_syntax','new_history','new_negation']})
  results={'experiment':'COG-WEIGHT-019','status':'posthoc frozen representation probe',
       'selection':'regularization chosen on 017 development ONLY',
       'notes':['No original model weights updated.','Test panels previously examined in earlier cycles; not an untouched confirmation.',
         'Probe knows synthetic binary support labels at training time; not an unlabeled learned verifier.'],
       'data_hashes':{'train':r.sha(raw['train']),'dev':r.sha(raw['dev']),
           'groups':{k:r.sha(v) for k,v in groups.items()}},'seeds':{},'elapsed_seconds':None}
  rows=[]
  for seed in SEEDS:
    beginseed=time.monotonic();base,sha=r.load016(seed)
    train=check(base,raw['train']);dev=check(base,raw['dev']); packs={k:check(base,v) for k,v in groups.items()}
    # Probe model selection is confined to the original 017 dev panel.
    choices=[]
    for C in (0.001,.01,.1,1,10):
      scale,clf=binary_probe(train,C)
      dv=metrics(scale,clf,dev)
      choices.append((float(dv['candidate_auc']),C))
    C=max(choices,key=lambda q:q[0])[1]
    scale,clf=binary_probe(train,C)
    stats={k:metrics(scale,clf,v) for k,v in packs.items()}
    results['seeds'][str(seed)]={'parent_sha256':sha,'selected_C':C,'dev_candidate_auc':metrics(scale,clf,dev)['candidate_auc'],
     'test':stats,'elapsed_seconds':round(time.monotonic()-beginseed,3)}
    for name,z in stats.items():rows.append({'seed':seed,'group':name,**z})
    print('seed',seed,'C',C,'tests',[(k,round(z['presence_auc'],3) if z['presence_auc'] is not None else None,round(z['candidate_auc'],3)) for k,z in stats.items()],flush=True)
  results['elapsed_seconds']=round(time.monotonic()-begin,3)
  (HERE/'RESULTS_019.json').write_text(json.dumps(results,indent=2))
  with (HERE/'SUMMARY_019.csv').open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print('finished',results['elapsed_seconds'])
if __name__=='__main__':run()