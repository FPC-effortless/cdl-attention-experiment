"""Active and passive hypothesis elimination on a four-rule issue workflow."""
from itertools import product

INPUTS=list(product((0,1),repeat=2))
def env(rule,flags):
    a,b=flags
    return bool((a,b,a|b,a&b)[rule])

# Model truth tables use independently enumerated accepted patterns.
ACCEPT=[{(1,0),(1,1)},{(0,1),(1,1)},{(1,0),(0,1),(1,1)},{(1,1)}]

class Learner:
    def __init__(self,mode):
        self.mode=mode;self.possible={0};self.probes=0

    def update(self,flags,outcome):
        if self.mode=='frozen':return
        consistent={r for r in range(4) if (flags in ACCEPT[r])==outcome}
        self.possible=self.possible.intersection(consistent) or consistent

    def audit(self,query,available):
        if self.mode!='active' or not available:return
        self.possible=set(range(4))
        while len(self.possible)>1:
            def balance(pair):
                k=sum(pair in ACCEPT[r] for r in self.possible)
                return min(k,len(self.possible)-k)
            diagnostic=max(INPUTS,key=balance)
            self.update(diagnostic,query(diagnostic));self.probes+=1

    def decision(self,flags):
        votes=[flags in ACCEPT[r] for r in self.possible]
        return votes[0] if all(x==votes[0] for x in votes) else None

def run(seed):
    import random
    rng=random.Random(seed); out=[]
    for silent in (False,True):
        for service in ('available','missing'):
            for episode in range(12):
                old=0;new=rng.choice([1,2,3]);shift=rng.randrange(9,20)
                fs=[rng.choice([(0,0),(1,1)] if silent else INPUTS) for i in range(48)]
                for mode in ('active','passive','frozen'):
                    m=Learner(mode); errors=uncertain=0; delay=None; start=0
                    for i,f in enumerate(fs):
                        hidden=new if i>=shift else old
                        if i==shift:start=m.probes
                        if i%8==0:m.audit(lambda p:env(hidden,p),service=='available')
                        value=m.decision(f)
                        if i>=shift:
                            errors+=int(value is not None and value!=env(hidden,f))
                            uncertain+=int(value is None)
                        m.update(f,env(hidden,f))
                        if i>=shift and m.possible=={new} and delay is None:delay=i-shift
                    probes=m.probes-start
                    out.append(dict(episode=episode,silent=silent,service=service,mode=mode,delay=delay,
                                    errors=errors,unknown=uncertain,probes=probes,horizon=48-shift,
                                    cost=5*probes+20*errors+uncertain))
    return out
