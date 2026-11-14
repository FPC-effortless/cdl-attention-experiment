import random,json
from model_source import encode,history_map,VOCAB,CUTOFF
TEMPLATES=('overwrite','delete_root','delete_alias','detach_alias','absence_snapshot')
KEYS=tuple(f'r{i}' for i in range(4))

def generate():
    rng=random.Random(520011);rows=[]
    for i in range(800):
        root,alias,spare,unrelated=rng.sample(KEYS,4);order=rng.sample(KEYS,4)
        old=rng.sample(range(8),2);new=rng.randrange(8);other={k:rng.randrange(8) for k in KEYS};own=rng.randrange(8)
        template=TEMPLATES[(i//2)%5];op=('copy','link')[i%2];worlds=[]
        for value in old:
            initial={**other,root:value}
            events=[{'op':'set','key':k,'value':f'v{initial[k]}'} for k in order]
            events += [{'op':op,'key':alias,'source':root},{'op':'set','key':root,'value':f'v{new}'}]
            if template in ('delete_root','absence_snapshot'):events.append({'op':'delete','key':root})
            if template=='delete_alias':events.append({'op':'delete','key':alias})
            if template=='detach_alias':events.append({'op':'set','key':alias,'value':f'v{own}'})
            if template=='absence_snapshot':events.append({'op':'copy','key':spare,'source':root})
            worlds.append({'schema':'woyengi.lifecycle.v1','keys':list(KEYS),'events':events,'composed_keys':[alias,spare]})
        pair={'pair_id':f'pair{i:04d}','template':template,'focal_op':op,'root':root,'alias':alias,'spare':spare,'unrelated':unrelated,'new':f'v{new}','old':[f'v{x}' for x in old],'worlds':worlds}
        rows.append({'record_id':pair['pair_id'],'episode_id':pair['pair_id'],'subject_id':pair['pair_id'],'observed_at':CUTOFF,'available_at':CUTOFF,'observation':{'pair':pair},'action':None,'outcome':{'status':'unknown','verified':False}})
    return rows

def validate(rows,replay):
    if len(rows)!=800:raise ValueError('pair count')
    ids=set();streams=set()
    for row in rows:
        p=row['observation']['pair'];identity=p['pair_id']
        if identity in ids:raise ValueError('duplicate pair')
        ids.add(identity)
        for w in p['worlds']:
            signature=json.dumps(w['events'],sort_keys=True)
            if signature in streams:raise ValueError('duplicate token history')
            streams.add(signature)
            if history_map(w)!=replay(w):raise ValueError('oracle mismatch')
            prefix={**w,'events':w['events'][:6]}
            if replay(prefix)[p['root']]!=p['new']:raise ValueError('wrong invariant target')
            for q in KEYS:
                x=encode(w,q)
                if len(x['ids'])!=27 or x['ids'][15:18]!=[VOCAB['set'],VOCAB[p['root']],VOCAB[p['new']]]:raise ValueError('SET boundary')
        a,b=p['worlds'];diff=[j for j,(x,y) in enumerate(zip(a['events'],b['events'])) if x!=y]
        if len(diff)!=1 or diff[0]>3 or a['events'][diff[0]]['key']!=p['root']:raise ValueError('pair not old-only')
    return {'pairs':len(ids),'distinct_histories':len(streams),'templates':list(TEMPLATES)}

def example(world,key):
    answer=history_map(world)[key]
    return {**encode(world,key),'answer':VOCAB['unknown' if answer is None else answer]}
