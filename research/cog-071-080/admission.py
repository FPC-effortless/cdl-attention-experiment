"""Induction from full utterances, exact semantic memoization and targeted causal tests."""
from collections import OrderedDict,defaultdict
from pathlib import Path
import random,re,json,os
import numpy as np
from causal import env,INPUTS,ACCEPT,Learner
from language import Grounder,render,TARGETS
from transfer import BANKS,novel_layout,final_cal
from tracker import run as session

ROOT=Path(__file__).resolve().parent
def grams(text):
    words=re.findall(r'[a-z]+',text.lower())
    return {tuple(words[i:i+n]) for n in range(1,5) for i in range(len(words)-n+1)}

class Inducer:
    def __init__(self,observations):
        evidence=defaultdict(lambda:defaultdict(set));self.dictionary={}
        for text,before,after,root,role in observations:
            delta=np.flatnonzero(before!=after)
            if len(delta)!=1:continue
            label=int(delta[0])
            for phrase in grams(text):evidence[phrase][role,label].add(root)
        for phrase,table in evidence.items():
            labels={v for _,v in table}
            if len(labels)!=1:continue
            label=next(iter(labels))
            if len(table['train',label])>=2 and table['test',label]-table['train',label]:
                self.dictionary[phrase]=label
    def apply(self,text,p):
        votes={self.dictionary[g] for g in grams(text) if g in self.dictionary}
        if len(votes)==1:
            out=np.full(len(p),.01/(len(p)-1));out[votes.pop()]=.99;return out
        return p

def observations():
    rows=[]
    for a,phrases in enumerate(BANKS['transfer']):
        for phrase in phrases:
            for j,(root,role) in enumerate([('qa-a','train'),('qa-b','train'),('reference','test')]):
                text=render(phrase,TARGETS[(a+j)%4][j],j)
                before=np.arange(8)+j
                if role=='train':after=before.copy();after[a]+=1
                else:after=np.array([v+(i==a) for i,v in enumerate(before)])
                rows.append((text,before,after,root,role))
    return rows

class Memo:
    def __init__(self,expanded):
        self.base=Grounder(expanded);self.cache=OrderedDict()
    def score(self,texts):
        missing=list(dict.fromkeys(t for t in texts if t not in self.cache))
        local={t:self.cache[t] for t in texts if t in self.cache}
        if missing:
            pp,qq=self.base.score(missing)
            local.update({t:(a,b) for t,a,b in zip(missing,pp,qq)})
        for t in texts:
            self.cache[t]=local[t];self.cache.move_to_end(t)
            if len(self.cache)>128:self.cache.popitem(last=False)
        a,b=zip(*(local[t] for t in texts));return np.array(a),np.array(b)

def targeted(seed):
    rng=random.Random(seed);out=[]
    for confounded in (False,True):
        for available in (False,True):
            for i in range(24):
                shift=rng.randrange(10,23);rule=rng.choice([1,2,3]);flags=[rng.choice([(0,0),(1,1)] if confounded else INPUTS) for _ in range(48)]
                for mode in ('frozen','passive','regular','demand'):
                    m=Learner('active' if mode in ('regular','demand') else mode)
                    deadline=8;probes=errors=unknown=0;diagnosis=None
                    for t,f in enumerate(flags):
                        h=rule if t>=shift else 0
                        if mode=='regular' and t%8==0:
                            m.possible=set(range(4));m.audit(lambda x:env(h,x),available)
                        if mode=='demand' and t>=deadline and (f[0]!=f[1] or t==40):
                            m.possible=set(range(4));deadline=t+8
                        if mode=='demand' and t==40:
                            m.possible=set(range(4));m.audit(lambda x:env(h,x),available)
                        if mode=='demand' and m.decision(f) is None and available:
                            m.update(f,env(h,f));m.probes+=1
                        if t==40:diagnosis=next(iter(m.possible)) if len(m.possible)==1 else None
                        guess=m.decision(f)
                        unknown+=int(guess is None);errors+=int(guess is not None and guess!=env(h,f))
                        m.update(f,env(h,f))
                    cost=m.probes*5+errors*20+unknown+(1 if diagnosis is None else 20*int(diagnosis!=rule))
                    out.append(dict(i=i,confounded=confounded,available=available,mode=mode,
                                    right=diagnosis==rule,unknown_diagnosis=diagnosis is None,
                                    cost=cost,probes=m.probes,wrong=errors,unknown=unknown))
    return out

def main():
    cases=[];causal={};ds=observations();ind=Inducer(ds)
    assert ind.dictionary
    for seed in (8189,8191,8209):
        arms=['raw','supported_exact','supported_cache','memo_exact','memo_cache'];random.Random(seed).shuffle(arms)
        for arm in arms:
            opts=dict(factory=Memo if arm.startswith('memo') else Grounder,layout=novel_layout,calibrator=final_cal)
            if arm!='raw':opts.update(inducer=lambda:Inducer(observations()),support_work=sum(len(grams(r[0])) for r in ds))
            s=session(seed,'transfer','full' if 'cache' in arm else 'primitive',opts=opts);s['arm']=arm;cases.append(s)
        causal[seed]=targeted(seed)
    (Path(os.getenv('COG_RESULT_OUT',ROOT/'results'))/'admission.json').write_text(json.dumps(dict(sessions=cases,causal=causal,
        dictionary={' '.join(k):v for k,v in sorted(ind.dictionary.items())}),separators=(',',':')))
if __name__=='__main__':main()
