"""Profile exact reusable workflow plans against the pinned COG-066 planner."""
from collections import Counter
from time import perf_counter_ns
from vendor.packet.domain import World, initial, step, done
from vendor.packet.planning import plan,acquire

def layout(ticket,op,epoch):
    # Three checklist lines must be visited at the proper workflow desks.
    return World(((ticket,(op+epoch)%4),((op+ticket)%4,),((ticket+epoch+1)%4,)),(2,1,1),('owner','qa','release'))

def checked(w,path):
    s=initial(w)
    for a in path:
        move=step(w,s,a)
        if move is None:return False
        s=move[0]
    return done(w,s)

class Planner:
    def __init__(self,mode):
        self.mode=mode;self.cache={};self.seen=Counter();self.options=()
        self.training=[]

    def reset(self):
        n=len(self.cache)
        self.cache.clear();self.seen.clear();self.training.clear();self.options=()
        return n

    def solve(self,w):
        # World includes all preconditions and no blocked checklist desk in this workload.
        units=1
        if self.mode=='cache' and w in self.cache:
            path=self.cache[w]
            if not checked(w,path):raise AssertionError('cached plan validation failed')
            return path,units+len(path),True
        result=plan(w,initial(w),options=self.options)
        path=result.path
        assert checked(w,path)
        units+=result.attempted_primitives+len(path)
        self.seen[w]+=1
        if self.mode=='cache' and self.seen[w]==2 and len(self.cache)<8:
            # Charge storage, validation and admission against two prior search costs.
            build=len(path)+2
            if units>2*build:
                self.cache[w]=path; units+=build
        if self.mode=='chunks' and len(self.training)<24:
            self.training.append(dict(verified=True,path=path,source_family=str(w.bays)))
            if len(self.training)==24:
                self.options=acquire(self.training);units+=sum(len(r['path'])*3 for r in self.training)
        return path,units,False

def economic(seed):
    import random
    rng=random.Random(seed);result={}
    for hot in (True,False):
        jobs=[(1,0) if hot and rng.random()<.8 else (rng.randrange(4),rng.randrange(8)) for _ in range(144)]
        arms=['primitive','chunks','cache'];rng.shuffle(arms)
        for mode in arms:
            start=perf_counter_ns();engine=Planner(mode);work=hits=0;costs=[]
            for i,(t,a) in enumerate(jobs):
                if i%48==0:work+=engine.reset()
                w=layout(t,a,i//48);p,u,h=engine.solve(w);work+=u;hits+=h
                state=initial(w);cost=0
                for move in p:
                    state,v=step(w,state,move);cost+=v
                assert done(w,state)
                costs.append(cost)
            result[str(hot)+'/'+mode]=dict(work=work,elapsed_ns=perf_counter_ns()-start,hits=hits,costs=costs)
        assert result[str(hot)+'/primitive']['costs']==result[str(hot)+'/cache']['costs']==result[str(hot)+'/chunks']['costs']
    return result
