"""New public DEVELOPMENT streams. No new operation or identity tokens."""
import hashlib
import json
import random
from collections import Counter
DEPTHS = (1, 4, 16, 64, 256)
REGIMES = ('all_changed', 'one_changed')
TOKENS = ('<pad>', '<bos>', 'set', ';', 'ask', '?', '<nop>') + tuple(f'r{i}' for i in range(4)) + tuple(f'v{i}' for i in range(8))
VOCAB = {t:i for i,t in enumerate(TOKENS)}

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def canonical(writes):
    names={}
    return [[names.setdefault(k,len(names)),v] for k,v in writes]

def generate(forbidden):
    rng=random.Random(131126)
    worlds=[]
    for block in range(8):
        for attempt in range(10000):
            order=rng.sample(range(4),4)
            initial=rng.sample(range(8),4)
            increments=[rng.randrange(1,8) for _ in range(256)]
            proposed=[]
            for shift in range(8):
                for regime in REGIMES:
                    writes=[[k,(v+shift)%8] for k,v in zip(order,initial)]
                    current=dict(writes)
                    for i,inc in enumerate(increments):
                        k=order[i%4] if regime=='all_changed' else order[0]
                        v=(current[k]+inc)%8
                        writes.append([k,v]); current[k]=v
                        if i+1 in DEPTHS:
                            proposed.append({'family':f'b{block}-s{shift}', 'block':block,
                                'regime':regime,'updates':i+1,'writes':[w[:] for w in writes]})
            if not any(digest(canonical(w['writes'])) in forbidden for w in proposed):
                worlds.extend(proposed)
                break
        else: raise RuntimeError('Could not exclude TRAIN histories')
    return worlds

def encode(world,query):
    words=['<bos>']; dest=[-100]; edit=[0]
    current={}; prior={}; changed=set()
    for j,(key,value) in enumerate(world['writes']):
        if key in current:
            if value==current[key]: raise ValueError('overwrite must change value')
            prior[key]=current[key]; changed.add(key)
        current[key]=value
        words += ['set',f'r{key}',f'v{value}',';']
        dest += [-100,-100,key,-100]; edit += [0,0,1,0]
    words += ['<nop>']*4+['ask',f'r{query}','?']
    dest += [-100]*6+[query]; edit += [0]*7
    return {'input_ids':[VOCAB[t] for t in words], 'answer':VOCAB[f'v{current[query]}'],
        'destination_targets':dest, 'edit_targets':edit,
        'prior':None if query not in prior else VOCAB[f'v{prior[query]}'],
        'changed':query in changed,'query':query,'family':world['family'],'block':world['block'],
        'updates':world['updates'],'regime':world['regime']}

def validate(worlds,forbidden):
    if len(worlds)!=640: raise ValueError('wrong world count')
    for regime in REGIMES:
        for depth in DEPTHS:
            group=[w for w in worlds if (w['regime'],w['updates'])==(regime,depth)]
            if len(group)!=64 or len({w['family'] for w in group})!=64: raise ValueError('families')
            rows=[encode(w,q) for w in group for q in range(4)]
            if Counter(r['answer'] for r in rows)!=Counter({VOCAB[f'v{i}']:32 for i in range(8)}): raise ValueError('balance')
            for w in group:
                if digest(canonical(w['writes'])) in forbidden: raise ValueError('TRAIN overlap')
                if len(w['writes']) != 4+depth: raise ValueError('depth')
                if set(k for k,v in w['writes'][:4]) != set(range(4)): raise ValueError('identity')
    return {'worlds':640,'queries':2560,'paired_value_shift_blocks':8,'families_per_panel':64,
        'independent_data_unit':'base block; eight value shifts are correlated',
        'panels':10,'exact_answer_balance':True,'train_complete_history_overlap':0}
