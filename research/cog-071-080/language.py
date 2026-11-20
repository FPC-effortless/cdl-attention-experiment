"""Classifiers receive text only. Data construction and observation roots are explicit."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
from time import perf_counter_ns
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BANKS = {
 'train': [
 ['close','mark resolved','finish the work on'],['reopen','bring back','mark unresolved'],
 ['assign','send to an owner','allocate'],['defer','put off','delay'],
 ['prioritize','make urgent','raise priority for'],['label','add a tag to','categorize'],
 ['archive','put in storage','file away'],['publish','announce','release notes for']],
 'cal':[['close out','finish up'],['back in progress','re-open'],['find an owner for','assign ownership of'],
        ['defer work on','delay processing'],['urgent handling of','move up'],['tag please','label as usual'],
        ['move to storage','archive entry'],['announce changes for','publish entry']],
 'primary':[['wrap up','this can be closed'],['give another look to','it belongs back on the board'],
            ['someone should own','hand off'],['can wait','leave for later'],['jump the queue for','this is pressing'],
            ['stick a tag on','needs a category'],['shelve for good','retire the record for'],
            ['tell everyone about','ship the announcement for']],
 'rep':[['call it done','put a lid on'],['resurrect','that fix did not stick for'],
        ['make somebody responsible for','route to a maintainer'],['back burner for','park until next week'],
        ['expedite','front of the line for'],['sort by topic','attach a marker to'],
        ['move into long term storage','box up the history of'],['make public','post the release for']]}
TARGETS=[['authentication','login','sign in'],['billing','invoice','charges'],['storage','disk','files'],
         ['notifications','alerts','messages']]
ALIASES=['sundown','rekindle','baton','raincheck','fastlane','sticker','vault','broadcast']
def render(phrase,target,layout):
    return [phrase+' '+target,'about '+target+' I think we should '+phrase,
            target+' — '+phrase+' please','could you '+phrase+' the '+target+' ticket'][layout]

class TextModel:
    def __init__(self,docs,y,expanded):
        self.vectors=[TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)]
        if expanded:
            self.vectors.append(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True))
        x=hstack([v.fit_transform(docs) for v in self.vectors],format='csr')
        self.model=LogisticRegression(C=12,max_iter=300,random_state=71).fit(x,y)

    def score(self,texts):
        return self.model.predict_proba(hstack([v.transform(texts) for v in self.vectors],format='csr'))

class Grounder:
    def __init__(self,expanded):
        docs=[];ops=[];targets=[]
        for a,t,k,l in itertools.product(range(8),range(4),range(3),range(4)):
            for target in TARGETS[t]:
                docs.append(render(BANKS['train'][a][k],target,l));ops.append(a);targets.append(t)
        self.op=TextModel(docs,ops,expanded);self.target=TextModel(docs,targets,expanded)
        self.cache={}

    def score(self,texts):
        return self.op.score(texts),self.target.score(texts)

class Lexicon:
    def __init__(self):
        self.bindings={}; self.provenance={}

    def learn(self,examples):
        by=defaultdict(lambda:defaultdict(set))
        for phrase,root,observed,split in examples:
            by[phrase][split,observed].add(root)
        for phrase,counts in by.items():
            values={v for split,v in counts}
            if len(values)!=1:continue
            v=next(iter(values));a=counts['acquire',v];b=counts['verify',v]
            if len(a)>=2 and b-a:
                self.bindings[phrase]=v
                self.provenance[phrase]=sorted(a|b)

    def apply(self,text,p):
        matches={self.bindings[w] for w in text.lower().split() if w in self.bindings}
        if len(matches)==1:
            q=np.full(len(p),.01/(len(p)-1));q[matches.pop()]=.99
            return q
        return p

def outcome_examples():
    # Executed status transitions reported by simulated QA and replay witnesses.
    examples=[]
    for a,phrase in enumerate(ALIASES):
        for root,split in [('qa-a','acquire'),('qa-b','acquire'),('reference','verify')]:
            # Index of the changed workflow field is read from the transition,
            # never passed to the learner by the phrase generator.
            old=np.arange(8) + {'qa-a':10,'qa-b':30,'reference':70}[root]
            if split=='verify':
                new=np.array([v+int(j==a) for j,v in enumerate(old)])
            else:
                new=old.copy();new[a]+=1
            changed=np.flatnonzero(new!=old)
            assert len(changed)==1
            examples.append((phrase,root,int(changed[0]),split))
    examples+= [('echo','shared',3,'acquire')]*12+[('echo','shared',3,'verify'),
               ('disputed','a',2,'acquire'),('disputed','b',2,'acquire'),('disputed','c',5,'verify')]
    return examples

@dataclass
class Case:
    a:int
    t:int
    messages:list
    permitted:bool
    explicit:bool
    reply:bool
    available:bool
    flags:tuple
    special:str

def cases(seed,bank,n=192):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        a=int(rng.integers(8));t=int(rng.integers(4));style=int(rng.integers(4))
        def message(op,ticket,phrase=None):
            return render(str(phrase or rng.choice(BANKS[bank][op])),str(rng.choice(TARGETS[ticket])),style)
        special=['plain','echo','correct','implicit','skill','conflict','plain','plain'][i%8]
        initial=message(a,t)
        if special=='correct':initial=message((a+3)%8,(t+1)%4)
        if special=='implicit':initial='take care of that thing'
        if special=='skill':initial=message(a,t,ALIASES[a])
        msgs=[('user',False,initial)]
        if special=='echo':msgs += [('user',False,initial)]*2
        if special=='correct':msgs.append(('user',True,message(a,t)))
        if special=='conflict':msgs += [('mirror',False,message((a+1)%8,t)),('user',True,message(a,t))]
        if i%3==0:msgs.append(('qa',False,message(a,t)))
        msgs.append(('silence',False,''))
        bit=int(rng.integers(2))
        flags=(bit,bit) if i%7 else (bit,1-bit)
        rows.append(Case(a,t,msgs,rng.random()>.13,rng.random()>.28,rng.random()>.28,rng.random()>.22,flags,special))
    return rows

def top(op,target):
    scores=np.outer(op,target).ravel()
    return np.argsort(-scores,kind='stable')

def calibrate(model):
    ds=cases(71001,'cal',512);p,q=model.score([c.messages[0][2] for c in ds])
    hits=np.array([top(a,b)[0]==4*c.a+c.t for c,a,b in zip(ds,p,q)])
    conf=p.max(1)*q.max(1)
    bins=np.linspace(0,1,11)
    return [float((hits[(conf>=lo)&(conf<hi)].sum()+1)/(((conf>=lo)&(conf<hi)).sum()+2)) for lo,hi in zip(bins,bins[1:])]
