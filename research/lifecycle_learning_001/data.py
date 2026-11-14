"""Five known one-hop mechanisms; no independent-template generalization claim."""
from itertools import permutations
import random
import json

TOKENS = ['<pad>', 'set', 'copy', 'link', 'delete', 'ask', '_'] + [f'r{i}' for i in range(4)] + [f'v{i}' for i in range(8)] + ['unknown']
VOCAB = {t:i for i,t in enumerate(TOKENS)}
TEMPLATES = ('overwrite','delete_gap','recreate_snapshot','detach','delete_alias')
HEADS = ('first_address','second_address','destination','use_link','value_edit','link_edit')
CUTOFF = '2026-11-14T00:00:00Z'


def generate():
    rng = random.Random(140011)
    assignments = []
    while len(assignments) < 128:
        item = (tuple(rng.sample(range(4),4)), tuple(rng.sample(range(8),4)))
        if item not in assignments:
            assignments.append(item)
    rows=[]
    for group,(roles,values) in enumerate(assignments):
        a,b,c,u = [f'r{i}' for i in roles]
        old,new,other,own = [f'v{i}' for i in values]
        for template in TEMPLATES:
            for operation in ('copy','link'):
                events=[{'op':'set','key':a,'value':old}, {'op':'set','key':c,'value':own},
                        {'op':'set','key':u,'value':other}, {'op':operation,'key':b,'source':a}]
                if template == 'overwrite':
                    events += [{'op':'set','key':a,'value':new}]
                elif template == 'delete_gap':
                    events += [{'op':'delete','key':a}]
                elif template == 'recreate_snapshot':
                    events += [{'op':'delete','key':a},{'op':'copy','key':c,'source':a},
                               {'op':'set','key':a,'value':new}]
                elif template == 'detach':
                    events += [{'op':'set','key':b,'value':own},{'op':'set','key':a,'value':new}]
                else:
                    events += [{'op':'delete','key':b},{'op':'set','key':a,'value':new}]
                memory={'schema':'woyengi.lifecycle.v1','keys':[f'r{i}' for i in range(4)],
                        'events':events,'composed_keys':[b,c]}
                identity=f'g{group:03d}-{template}-{operation}'
                rows.append({'record_id':identity,'episode_id':identity,'subject_id':f'g{group:03d}',
                    'observed_at':CUTOFF,'available_at':CUTOFF,
                    'observation':{'memory':memory,'condition':{'template':template,'focal_op':operation,
                        'roles':{'source':a,'alias':b,'snapshot':c,'unrelated':u}}},
                    'action':None,'outcome':{'status':'unknown','verified':False}})
    # Unused value choices can produce identical streams across assignments.
    # Keep every connected assignment together, without counting these as new mechanisms.
    parent={r['subject_id']:r['subject_id'] for r in rows}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for row in rows:
        signature=json.dumps(row['observation']['memory']['events'],sort_keys=True,separators=(',',':'))
        subject=row['subject_id']
        if signature in seen:
            a,b=find(subject),find(seen[signature]);parent[max(a,b)]=min(a,b)
        else:seen[signature]=subject
    for row in rows:row['subject_id']='component-'+find(row['subject_id'])
    return rows


def history_map(memory):
    """Independent backward-time semantics, including COPY snapshots of UNKNOWN."""
    events=memory['events']
    def read(key,end,seen):
        if (key,end) in seen:return None
        seen=seen|{(key,end)}
        for index in range(end-1,-1,-1):
            event=events[index]
            if event.get('key')!=key:continue
            op=event['op']
            if op=='set':return event['value']
            if op=='delete':return None
            if op=='copy':return read(event['source'],index,seen)
            if op=='link':return read(event['source'],end,seen)
        return None
    return {key:read(key,len(events),set()) for key in memory['keys']}


def encode(memory,query):
    ids=[];targets={h:[] for h in HEADS};targets['write_gate']=[]
    events=memory['events']+[{'op':'nop'}]*(8-len(memory['events']))+[{'op':'ask','key':query}]
    if len(events)!=9:raise ValueError('unsupported event length')
    for event in events:
        op=event['op']
        if op=='nop':tokens=['_']*3
        else:tokens=[op,event['key'],event.get('source',event.get('value','_'))]
        label={h:-100 for h in HEADS};label.update(value_edit=0,link_edit=0,write_gate=-100)
        for pos,token in enumerate(tokens):
            t=dict(label)
            if pos==2 and op!='nop':
                dest=int(event['key'][1:]);source=int(event.get('source',event['key'])[1:])
                t.update(first_address=source if op in ('copy','link') else dest,
                         second_address=source if op=='link' else dest,
                         destination=dest,use_link=0,value_edit=int(op!='link'),link_edit=1,
                         write_gate=1 if op!='link' else -100)
                if op=='ask':
                    t.update(first_address=dest,second_address=-100,destination=7,
                             use_link=1,value_edit=1,link_edit=0,write_gate=1)
            ids.append(VOCAB[token])
            for h in targets:targets[h].append(t[h])
    return {'ids':ids,**targets}


def examples(forge_examples):
    output=[]
    for row in forge_examples:
        memory=row['features']['memory'];condition=row['features']['condition'];answers=history_map(memory)
        for query in memory['keys']:
            output.append({'record_id':row['record_id'],'subject_id':row['subject_id'],
                'query':query,'template':condition['template'],'focal_op':condition['focal_op'],
                'role':next(k for k,v in condition['roles'].items() if v==query),
                'answer':VOCAB['unknown' if answers[query] is None else answers[query]],
                **encode(memory,query)})
    return output
