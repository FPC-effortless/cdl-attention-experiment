"""Replay genuine issue comments into separate execution and observation trust domains."""
from collections import Counter
import hashlib,json,math,os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
import joblib
import numpy as np
from scipy.sparse import hstack
from boundary import Authority,Ledger,Workspace
from tracker import provisional

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def main():
    start=perf_counter_ns()
    source=Path(os.environ['COG_CHECKPOINT_DIR'])/'issues/expanded.joblib'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==read(ROOT/'results/checkpoints.json')['restricted-github-expanded.joblib']
    vec,model=joblib.load(source)
    older=Path('/workspace/issue-history-study/recipe/dataset.json')
    fresh=Path('/workspace/issue-history-followup/recipe/dataset.json')
    evidence=read(ROOT/'results/authentic-issues.json')
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in (older,fresh)]==evidence['source']
    cal=[r for r in read(older) if r['split']=='validation' and r['strict']]
    rows=[r for r in read(fresh) if r['strict']]
    def scores(part):
        return model.predict_proba(hstack([v.transform([r['text'] for r in part]) for v in vec],format='csr'))[:,1]
    p=scores(cal);pos=sorted((v for v,r in zip(p,cal) if r['target']),reverse=True)
    cut=pos[math.ceil(.8*len(pos))-1]
    predictions=scores(rows)
    assert np.allclose(predictions,[r[4] for r in evidence['arms']['expanded']['rows']],rtol=0,atol=1e-12)
    replies={r[2]:r for r in read(ROOT.parent/'cog-attn-handoff/continuation/cog-051-060/results/confirmation.json')['rows']}
    stream=[];counter=Counter();traces=[]
    for row,prob in zip(rows,predictions):
        stream.append((row['stamp'],0,row['comment'],row,float(prob)))
        if prob>=cut and row['comment'] in replies:
            lag=replies[row['comment']][-1]
            stream.append((row['stamp']+lag,1,row['comment'],row,float(prob)))
    keys=[b'execution-only',b'observation-only'];observer=Authority(keys[1])
    with TemporaryDirectory() as tmp:
        # Each real issue has its own ledger namespace and its own staged workspaces.
        handles={};pending={}
        for t,delivery,cid,r,p in sorted(stream,key=lambda x:x[:3]):
            issue=r['issue']
            if not delivery:
                counter['comments']+=1
                if p<cut:continue
                if issue not in handles:
                    folder=Path(tmp)/str(len(handles));folder.mkdir()
                    handles[issue]=[Ledger(folder/str(j),key) for j,key in enumerate(keys)]
                execution,mirror=handles[issue];w=Workspace()
                a=int(bool(r['closed']))
                p0=provisional(w,execution,cid,0,a)
                assert execution.commit(p0,None)=='denied'
                counter['staged']+=1
                if cid in replies:pending[cid]=(w,p0)
                else:
                    w.discard('ticket');counter['unavailable']+=1
            else:
                execution,mirror=handles[issue];w,p0=pending.pop(cid)
                _,backing_issue,_,_,_,observed,lag=replies[cid]
                assert backing_issue==issue
                counter['responses']+=1;counter['observation_wait_seconds']+=lag
                if observed:
                    proposal=mirror.propose(str(cid),0,p0.op)
                    token=observer.issue(proposal,True,False)
                    assert execution.commit(proposal,token)=='denied'
                    assert mirror.commit(proposal,token)=='committed'
                    counter['mirrored']+=1;counter['mirror_discordant']+=int(not r['target'])
                    traces.append([issue,cid,r['stamp'],t,p0.op])
                w.discard('ticket')
        assert not pending
        for execution,mirror in handles.values():
            counter['executed']+=execution.db.execute('SELECT count(*) FROM journal').fetchone()[0]
            counter['mirror_journal']+=mirror.db.execute('SELECT count(*) FROM journal').fetchone()[0]
            execution.db.close();mirror.db.close()
    assert counter['executed']==0 and counter['mirror_journal']==counter['mirrored']
    out=dict(validation_threshold=cut,counts=counter,traces=traces,elapsed_ns=perf_counter_ns()-start)
    (ROOT/'results/issue-workflow.json').write_text(json.dumps(out))
if __name__=='__main__':main()
