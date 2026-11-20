"""Third-cohort semantic, causal and economic interventions."""
from collections import Counter
import hashlib,json,os
from pathlib import Path
from time import perf_counter_ns
import random
import numpy as np
from language import BANKS, Grounder, Lexicon, cases, render, TARGETS
from plans import Planner
from vendor.packet.domain import World
from causal import Learner

HERE=Path(__file__).resolve().parent
BANKS['transfer']=[['put this one to bed','consider it sorted'],['this needs another round','bring it out of retirement'],
                   ['pass the baton to someone','find a steward'],['push this to next cycle','leave on ice'],
                   ['bump it to the front','this deserves immediate attention'],['give it a topic marker','add the appropriate badge'],
                   ['move it to the permanent record','send to cold storage'],['let customers know','send the bulletin']]
HYP=['The issue should be closed as resolved.','The issue should be reopened.',
     'A person should be assigned to this issue.','Work on this issue should be postponed.',
     'The issue needs urgent priority.','A category label should be added.',
     'The record should be archived.','A public update should be announced.']
SEEDS=(8111,8123,8147)
def teacher():
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from scipy.special import softmax
    started=perf_counter_ns()
    model=Path(os.getenv('COG_NLI_DIR','/workspace/issue-history-study/recipe/nli'))
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model.iterdir()) if p.is_file()}
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
    net=ort.InferenceSession(str(model/'model_quint8_avx2.onnx'),opt,providers=['CPUExecutionProvider'])
    words=set()
    for seed,bank,n in [(71001,'cal',512)]+[(s,'transfer',192) for s in SEEDS]:
        for c in cases(seed,bank,n):
            words.update(m[2] for m in c.messages if m[2])
    docs=sorted(words);out=[];ids={x.name for x in net.get_inputs()}
    for i in range(0,len(docs),6):
        batch=docs[i:i+6]
        x=tok([s for s in batch for _ in HYP],HYP*len(batch),padding=True,return_tensors='np')
        v=softmax(net.run(None,{k:v for k,v in x.items() if k in ids})[0],axis=1)[:,1].reshape(-1,8)
        out.extend(v.tolist())
    (HERE/'results/teacher.json').write_text(json.dumps(dict(inputs=inputs,scores=dict(zip(docs,out)),
                                                           elapsed_ns=perf_counter_ns()-started)))
class Hybrid:
    def __init__(self,teacher_scores,use_teacher):
        self.base=Grounder(True);self.teacher=teacher_scores;self.weight=0
        if use_teacher:
            cal=cases(71001,'cal',512)
            texts=[c.messages[0][2] for c in cal]
            raw,_=self.base.score(texts)
            best=-1
            for w in (0,.5,1,2,4):
                self.weight=w;p=self.mix(texts,raw)
                count=sum(r.argmax()==c.a for r,c in zip(p,cal))
                if count>best:best=count;weight=w
            self.weight=weight
    def mix(self,docs,p):
        if not self.weight:return p
        mass=p*np.maximum(1e-8,np.array([self.teacher[t] for t in docs]))**self.weight
        return mass/mass.sum(1)[:,None]
    def score(self,texts):
        p,q=self.base.score(texts);return self.mix(texts,p),q

def final_cal(model):
    from tracker import infer
    ds=cases(71001,'cal',512);docs=[m[2] for c in ds for m in c.messages if m[2]]
    p,q=model.score(docs);buckets=[[] for _ in range(10)];j=0
    for c in ds:
        evidence=[]
        for speaker,rev,txt in c.messages:
            if txt:
                evidence.append(('user' if speaker=='mirror' else speaker,rev,p[j],q[j]));j+=1
        a,b=infer(evidence,'full')
        buckets[min(9,int(10*a.max()*b.max()))].append((a.argmax(),b.argmax())==(c.a,c.t))
    return [(sum(b)+1)/(len(b)+2) for b in buckets]

class Careful(Learner):
    def audit(self,query,available):
        if self.mode=='active' and not available:self.possible=set(range(4))
        super().audit(query,available)

def novel_layout(ticket,op,epoch):
    return World(((ticket,(op+epoch)%4,(op+1)%4),((op+ticket)%4,),((ticket+epoch+1)%4,)),(3,1,1),('owner','qa','release'))

class MeasuredPlanner(Planner):
    def __init__(self,seed):
        rng=random.Random(seed+710000)
        stream=[novel_layout(rng.randrange(4),rng.randrange(8),0) for _ in range(64)]
        cost={};times={};quality={};order=['primitive','cache'];rng.shuffle(order)
        for mode in order:
            candidate=Planner(mode);start=perf_counter_ns();work=0;paths=[]
            for world in stream:
                path,u,_=candidate.solve(world);work+=u;paths.append(path)
            cost[mode]=work;times[mode]=perf_counter_ns()-start;quality[mode]=paths
        mode='cache' if (quality['cache']==quality['primitive'] and cost['cache']<cost['primitive']
                         and times['cache']<times['primitive']) else 'primitive'
        super().__init__(mode);self.profile_work=sum(cost.values());self.profile=dict(cost=cost,times=times,admitted=mode)

def acquire_phrases():
    lex=Lexicon();obs=[]
    for a,phrases in enumerate(BANKS['transfer']):
        for phrase in phrases:
            for root,part in [('demonstrator','acquire'),('sandbox','acquire'),('inspector','verify')]:
                # Two transition implementations, with distinct initial conditions.
                state=np.arange(8)+len(root)
                if part=='verify':after=state+np.array([int(i==a) for i in range(8)])
                else:after=state.copy();after[a]+=1
                value=int(np.flatnonzero(state!=after)[0]);obs.append((phrase,root,value,part))
    lex.learn(obs);return lex

def run():
    from tracker import run as session
    archive=json.loads((HERE/'results/teacher.json').read_text())
    scores=archive['scores'];res=[]
    for seed in SEEDS:
        arms=['primitive','cache','repair','teacher','supported'];random.Random(seed).shuffle(arms)
        for arm in arms:
            options=dict(factory=lambda expanded:Hybrid(scores,arm in ('teacher','supported')),
                         calibrator=final_cal if arm in ('repair','teacher','supported') else None,
                         causal=Careful if arm in ('repair','teacher','supported') else Learner,
                         layout=novel_layout,planner=(lambda:MeasuredPlanner(seed)) if arm in ('repair','teacher','supported') else None,
                         extra_work=(len(scores)*8 if arm in ('teacher','supported') else 0),
                         external_ns=archive['elapsed_ns'] if arm in ('teacher','supported') else 0,
                         support_factory=acquire_phrases if arm=='supported' else None)
            r=session(seed,'transfer','primitive' if arm=='primitive' else 'full',opts=options)
            r['arm']=arm;res.append(r)
    (HERE/'results/transfer.json').write_text(json.dumps(res,separators=(',',':')))
if __name__=='__main__':
    import sys
    (teacher if sys.argv[1]=='teacher' else run)()
