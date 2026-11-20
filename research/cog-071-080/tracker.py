"""Persistent local issue workflow with source-aware reads and signed writes."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
import numpy as np
from boundary import Ledger,Authority,Workspace,OPERATIONS,HIGH,LOW
from causal import Learner, env
from language import Grounder, Lexicon, outcome_examples, calibrate, cases, top
from plans import Planner,layout,checked
VENDOR=Path(__file__).resolve().parent/'vendor'
sys.path.insert(0,str(VENDOR))
from upstream.belief import DialogueState, HOLD, EMPTY

ARMS=('full','primitive','chunks','word','no_lineage','pr73','always_ask','confidence',
      'no_stage','passive','frozen','no_skill')
STAGES=('setup','grounding','belief','planning','clarification','approval','probing',
        'staging','rollback','execution','verification','invalidation','calibration')
class Cost:
    def __init__(self):
        self.ns=Counter({k:0 for k in STAGES});self.work=self.ns.copy()

    def call(self,stage,units,fn,*args):
        t=perf_counter_ns()
        result=fn(*args)
        self.ns[stage]+=perf_counter_ns()-t;self.work[stage]+=units
        return result

def infer(evidence,arm):
    if arm=='pr73':
        labels={'op':[str(i) for i in range(8)]+[HOLD,EMPTY],
                'ticket':[str(i) for i in range(4)]+[HOLD,EMPTY]}
        s=DialogueState(labels,dict(op=2,ticket=2),method='soft',power=2,hold_weight=.25)
        try:
            for turn,(root,revision,pp,qq) in enumerate(evidence):
                s.observe(turn,[np.r_[p*.98,.02,0.] for p in (pp,qq)])
            return [v[:-2]/v[:-2].sum() for v in s.mass]
        finally:s.close()
    by={}
    for i,(root,revision,p,q) in enumerate(evidence):
        if revision and arm!='no_lineage':by.clear()
        key=i if arm=='no_lineage' else root
        by.setdefault(key,(p,q))
    out=[]
    for side in zip(*by.values()):
        mass=np.prod(np.maximum(side,1e-10),axis=0);out.append(mass/mass.sum())
    return out

def provisional(w,db,i,t,a):
    p=db.propose(str(i),int(t),OPERATIONS[int(a)])
    w.stage('ticket',(),p);w.stage('activity',('ticket',),p)
    return p

def run(seed,bank,arm,n=192,opts=None):
    opts=opts or {}
    allstart=perf_counter_ns();meter=Cost()
    model=meter.call('setup',1152,opts.get('factory',Grounder),arm!='word')
    cal=meter.call('calibration',512,opts.get('calibrator') or calibrate,model)
    meter.work['setup']+=opts.get('extra_work',0)
    support=meter.call('setup',48,opts['support_factory']).bindings if opts.get('support_factory') else {}
    lex=Lexicon()
    if arm!='no_skill':
        meter.call('setup',len(outcome_examples()),lex.learn,outcome_examples())
    trials=cases(seed,bank,n); rows=[]
    key=hashlib.sha256(str(seed).encode()).digest()
    authority=Authority(key);epoch=0
    learner=opts.get('causal',Learner)(arm if arm in ('frozen','passive') else 'active')
    planner=Planner('primitive' if arm=='primitive' else 'chunks' if arm=='chunks' else 'cache')
    if opts.get('planner'):
        planner=meter.call('planning',0,opts['planner']);meter.work['planning']+=planner.profile_work
    with TemporaryDirectory() as d:
        db=meter.call('setup',1,Ledger,Path(d)/'tracker.db',key)
        for i,c in enumerate(trials):
            before=sum(meter.work.values()); start=perf_counter_ns()
            if i in (64,128):
                epoch+=1
                # Public schema epoch changes at 64, hidden gate changes silently at 128.
                if i==64:
                    meter.call('invalidation',1,db.db.execute,'UPDATE policy SET epoch=epoch+1')
                    meter.work['invalidation']+=meter.call('invalidation',1,planner.reset)
            hidden=(0,(seed%3)+1,((seed+1)%3)+1)[epoch]
            if i%8==0:
                prev=learner.probes;meter.call('probing',1,learner.audit,lambda f:env(hidden,f),c.available)
                meter.work['probing']+=5*(learner.probes-prev)
            observed=[m for m in c.messages if m[2]]
            ps,qs=meter.call('grounding',len(observed),model.score,[m[2] for m in observed])
            first=meter.call('grounding',1,top,ps[0],qs[0])
            workspace=Workspace()
            if arm!='no_stage':
                meter.call('staging',2,provisional,workspace,db,i,first[0]%4,first[0]//4)
            beliefs=[]
            for (speaker,revision,txt),p,q in zip(observed,ps,qs):
                for phrase,value in support.items():
                    if phrase in txt:
                        p=np.full(8,.01/7);p[value]=.99;break
                beliefs.append(('user' if speaker=='mirror' else speaker,revision,
                                meter.call('grounding',1,lex.apply,txt,p),q))
            op,tgt=meter.call('belief',len(beliefs),infer,beliefs,arm)
            a=int(op.argmax());t=int(tgt.argmax())
            pre=(a,t);confidence=float(op[a]*tgt[t]);prob=cal[min(9,int(confidence*10))]
            # Separate calibration estimate from authorization. Ask only about the less certain slot.
            impact=30 if OPERATIONS[a] in HIGH else 6 if OPERATIONS[a] in LOW else 15
            should_ask=arm=='always_ask' or ((1-prob)*impact>3 if arm!='confidence' else confidence<.85)
            asked=None
            if should_ask:
                asked='operation' if op.max()<tgt.max() else 'ticket'
                reply=meter.call('clarification',20,lambda: (c.a if asked=='operation' else c.t) if c.reply else None)
                if reply is not None:
                    if asked=='operation':a=reply
                    else:t=reply
            rolled=0
            if workspace.nodes and (first[0]//4,first[0]%4)!=(a,t):
                rolled=meter.call('rollback',2,workspace.discard,'ticket')
            if not workspace.nodes:
                meter.call('staging',2,provisional,workspace,db,i,t,a)
            proposal=workspace.prepared('activity')
            w=opts.get('layout',layout)(t,a,db.epoch())
            path,work,hit=meter.call('planning',0,planner.solve,w);meter.work['planning']+=work
            valid=meter.call('execution',len(path),checked,w,path)
            guess=learner.decision(c.flags)
            # The independently evaluated workflow admission service enforces current gate and scope.
            actual=env(hidden,c.flags)
            allow=c.permitted and actual
            received=(not should_ask or c.reply or OPERATIONS[a] in LOW)
            approved=False;status='staged'
            if guess is True and valid and received:
                token=meter.call('approval',10,authority.issue,proposal,allow,c.explicit)
                approved=token is not None
                status=meter.call('verification',2,db.commit,proposal,token)
            elif guess is None or not received:status='abstained'
            if status!='committed':
                rolled+=meter.call('rollback',2,workspace.discard,'ticket')
            # Delayed ordinary telemetry contains the observed gate output for this feature pair.
            meter.call('probing',1,learner.update,c.flags,actual)
            correct=(c.a,c.t)==(a,t);committed=status=='committed'
            unauthorized=committed and not (approved and actual and c.permitted and (OPERATIONS[a] not in HIGH or c.explicit))
            rows.append(dict(i=i,kind=c.special,gold=4*c.a+c.t,top5=first[:5].tolist(),
                             provisional=4*pre[0]+pre[1],proposal=4*a+t,status=status,authorized=approved,
                             high=OPERATIONS[a] in HIGH,requested_eligible=allow and (OPERATIONS[c.a] not in HIGH or c.explicit),
                             bad_permission=bool(unauthorized),semantic_error=bool(committed and not correct),
                             asked=asked,reply=c.reply if asked else None,rolled=rolled,hit=hit,
                             work=sum(meter.work.values())-before,ns=perf_counter_ns()-start))
        j=db.db.execute('SELECT COUNT(*) FROM journal').fetchone()[0]
        assert j==sum(r['status']=='committed' for r in rows)
        db.db.close()
    return dict(seed=seed,arm=arm,bank=bank,elapsed_ns=perf_counter_ns()-allstart,
                **(dict(teacher_charge_ns=opts.get('external_ns',0),mixture=getattr(model,'weight',0),
                        profile=getattr(planner,'profile',None)) if opts else {}),
                units=dict(meter.work),times=dict(meter.ns),rows=rows)

def run_panel(seeds,bank):
    result=[]
    for seed in seeds:
        arms=list(ARMS);random.Random(seed).shuffle(arms)
        for arm in arms:result.append(run(seed,bank,arm))
    return result
