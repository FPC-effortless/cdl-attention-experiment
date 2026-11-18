import hashlib,json
from pathlib import Path
from collections import Counter
from reader import Reader,WORDS,V,data,np
from reference028 import State,Op,wait,commit,cancel,read

ROOT=Path(__file__).parent
TRAIN=['choose {p} over {q}','ignore {q} and choose {p}','{p} rather than {q}']
FORMS={'familiar':TRAIN,'reordered':['over {q} choose {p}'],
 'cambridge':['he studied {p} in preference to {q} at university'],
 'netlanguages':['she stayed at {p} instead of going to {q}'],
 'fronted_control':['instead of going to {q} she stayed at {p}']}
# Only training vocabulary, plus explicit boundary and unknown symbols.
CUES=['START','END','UNK']+[w for w in V if w not in WORDS]
def token(w):
    return V.get(w,0)
def feats(words,pos):
    left=words[pos-1] if pos else 'START'
    right=words[pos+1] if pos+1<len(words) else 'END'
    vec=np.zeros(2*len(CUES))
    for j,w in enumerate([left,right]):
        vec[j*len(CUES)+CUES.index(w if w in CUES else 'UNK')]=1
    return vec

def fit(seed):
    rng=np.random.default_rng(seed);weights=rng.normal(0,.005,(2*len(CUES),2))
    for step in range(400):
        dx=[];yy=[]
        for j in range(16):
            p,q=rng.choice(WORDS,2,replace=False)
            text=rng.choice(TRAIN).format(p=p,q=q).split()
            dx.append(feats(text,text.index(p))-feats(text,text.index(q)))
        dx=np.array(dx)
        probs=1/(1+np.exp(-dx@weights))
        weights-=.15*dx.T@(probs-[1.,0.])/len(dx)
    return weights

def freeze():
    datasets={}
    for i,(cohort,forms) in enumerate(FORMS.items()):
        rng=np.random.default_rng(290100+i)
        support,_,t,_=data(290200+i,512,'train')
        # Recover episode classes from unambiguous supplied support bindings.
        rev={v:k for k,v in V.items()}
        cases=[]
        for si,targets in zip(support,t):
            a,b=targets
            def alias(slot):
                word=next(rev[int(k)] for k in si[2*slot] if k and rev[int(k)] in WORDS)
                return WORDS[2*(WORDS.index(word)//2)+rng.integers(2)]
            p,q=alias(a),alias(b);form=rng.choice(forms)
            cases.append(dict(s=si.tolist(),t=targets.tolist(),w=[form.format(p=p,q=q),form.format(p=q,q=p)]))
        datasets[cohort]=cases
    import gzip
    with gzip.open(ROOT/'panels.gz','wt') as f:json.dump(datasets,f)
    (ROOT/'design.json').write_text(json.dumps(dict(gate='>=95% both reference directions and both conditional executions per cohort and seed; ties abstain',
      seeds=[11,23,37],forms=FORMS,training=TRAIN,panel_sha256=hashlib.sha256((ROOT/'panels.gz').read_bytes()).hexdigest(),
      intervention='use excluded mention for retrieval, fixed text, support, committed state, operation and weights; counterbalance guard presence and COPY/LINK for every query',
      source='Cambridge Dictionary and Net Languages adapted frames; fronting is investigator-authored; no independent participant responses'),indent=2))
    return datasets

def choose(score):
    k=score.argmax(-1);return np.where(np.isclose(score,score.max(-1,keepdims=True),atol=1e-10,rtol=0).sum(-1)==1,k,-1)

def outcome(initial,op,slot,gate):
    present=bool(slot>=0 and read(initial.cells,slot) is not None)
    allow=slot>=0 and np.dot([1,1,present,present],gate)>.5
    staged=wait(initial,Op(op,9,8))
    after=commit(staged) if allow else cancel(staged)
    later=commit(wait(after,Op('set',8,91)))
    return allow,later.cells,read(later.cells,9)

def main():
    datasets=freeze();results=[];logs={}
    for seed in [11,23,37]:
        m=Reader(seed,True);ck=np.load(ROOT/f'recurrent-{seed}.npz');m.E,m.a,m.b=[ck[k] for k in ('E','a','b')]
        gate=np.load(ROOT/f'commit-{seed}.npz')['predicate']
        weights=fit(seed);np.save(ROOT/f'roles-{seed}.npy',weights)
        for cohort,cases in datasets.items():
            s=np.array([c['s'] for c in cases]);t=np.array([c['t'] for c in cases]);n=len(s)
            store=m.forward(s)[0].reshape(n,8,2,32).sum(2)
            texts=[w.split() for c in cases for w in c['w']]
            # OOV tokens skipped by the frozen recurrent encoder's mask; all words retained for role features.
            full=np.zeros((len(texts),max(map(len,texts))),int)
            mentions=[];picks=[];second=[];truth=[]
            for i,words in enumerate(texts):
                full[i,:len(words)]=list(map(token,words))
                ids=[j for j,w in enumerate(words) if w in WORDS]
                assert len(ids)==2
                z=np.stack([feats(words,j) for j in ids])@weights
                picks.append(choose(z[:,0]));second.append(choose(z[:,1]))
                mentions.append([token(words[j]) for j in ids])
                actual_class=next(k for k in s[i//2,2*t[i//2,i%2]] if k in [V[w] for w in WORDS])
                truth.append(next(j for j,w in enumerate(mentions[-1]) if (w-V[WORDS[0]])//2==(actual_class-V[WORDS[0]])//2))
            mentions=np.array(mentions);truth=np.array(truth)
            def address(q):return choose(q.reshape(n,2,32)@store.transpose(0,2,1)).reshape(-1)
            def bymention(idx):
                idx=np.array(idx);word=mentions[np.arange(len(idx)),np.maximum(0,idx)]
                # Normalize each selected mention into the existing support phrase.
                sentence=np.zeros((len(idx),6),int);sentence[:,0]=V['entry'];sentence[:,1]=word
                return np.where(idx>=0,address(m.forward(sentence)[0]),-1)
            arms={'full':address(m.forward(full)[0]),'role':bymention(picks),'swap':bymention(second),'oracle':bymention(truth),'first':bymention(np.zeros(n*2,int))}
            for arm,addr in arms.items():
                a=addr.reshape(n,2);ok=a==t
                action=np.ones((n,2),bool);abstain=a<0
                for i,(ta,tb) in enumerate(t):
                    for polarity in range(2):
                        cells=[None]*10;cells[[ta,tb][polarity]]=('v',4);cells[8]=('v',7);cells[9]=('v',3)
                        state=State(tuple(cells))
                        for j in range(2):
                            for op in ['copy','link']:
                                actual=outcome(state,op,int(a[i,j]),gate)
                                allow=t[i,j]==[ta,tb][polarity]
                                # Independently specified expected destination after the delayed source write.
                                dest=(('v',7) if op=='copy' else ('l',8)) if allow else ('v',3)
                                expected=list(cells);expected[8]=('v',91);expected[9]=dest
                                action[i,j]&=actual==(allow,tuple(expected),7 if allow and op=='copy' else 91 if allow else 3)
                r=dict(seed=seed,cohort=cohort,arm=arm,route=int(ok.sum()),pair=int(ok.all(1).sum()),action_pair=int(action.all(1).sum()),abstain=int(abstain.sum()),mention_ok=int((np.array(picks)==truth).sum()),excluded_ok=int((np.array(second)==1-truth).sum()))
                results.append(r);logs[f'{seed}_{cohort}_{arm}']=a
    np.savez_compressed(ROOT/'predictions.npz',**logs)
    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
