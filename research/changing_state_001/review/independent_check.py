"""Fresh-session stdlib review. No training, neural import or evaluation."""
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lifecycle import Event, CurrentState, history_query
from stress_data import generate, encode, TOKENS

def check(condition, message):
    if not condition: raise RuntimeError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    alphabet=[Event(op,k,a) for op in ('SET','LINK','COPY') for k in range(2) for a in range(2)]
    alphabet += [Event('DELETE',k) for k in range(2)] + [Event('NOP')]
    trace_count=0; comparisons=0
    for events in product(alphabet,repeat=4):
        state=CurrentState();prefix=[]
        for event in events:
            state.apply(event);prefix.append(event)
            before=dict(state.records)
            for key in range(3):
                check(state.query(key)==history_query(prefix,key),'lifecycle mismatch')
                comparisons+=1
            check(before==state.records,'query mutation')
        trace_count+=1
    worlds=generate(set()); checked=0
    for world in worlds:
        # Independently derive each answer and predecessor by parsing input tokens.
        for q in range(4):
            row=encode(world,q); words=[TOKENS[i] for i in row['input_ids']]
            check(words[0]=='<bos>' and words[-7:-3]==['<nop>']*4,'boundary')
            check(words[-3:]==['ask',f'r{q}','?'],'query grammar')
            state={};previous={};changed=set();position=1
            while words[position]=='set':
                _,name,value,end=words[position:position+4]
                check(end==';','write boundary')
                if name in state:previous[name]=state[name];changed.add(name)
                state[name]=value
                check(row['destination_targets'][position:position+4]==[-100,-100,int(name[1:]),-100],'destination alignment')
                check(row['edit_targets'][position:position+4]==[0,0,1,0],'admission alignment')
                position+=4
            check(position==len(words)-7,'unexpected extra tokens')
            check(TOKENS[row['answer']]==state[f'r{q}'],'answer mismatch')
            check((None if row['prior'] is None else TOKENS[row['prior']])==previous.get(f'r{q}'),'predecessor mismatch')
            check(row['changed']==(f'r{q}' in changed),'change label')
            checked+=1
    for depth in (1,4,16,64,256):
        for regime in ('all_changed','one_changed'):
            panel=[w for w in worlds if w['updates']==depth and w['regime']==regime]
            check(len(panel)==64,'panel count')
            counts=Counter(encode(w,q)['answer'] for w in panel for q in range(4))
            check(len(counts)==8 and set(counts.values())=={32},'imbalance')
            check(len({w['block'] for w in panel})==8,'base block count')
    reviewed={p.name:sha(p) for p in ROOT.iterdir() if p.suffix in {'.py','.md'}}
    return {'status':'PASS','trace_count':trace_count,'prefix_query_comparisons':comparisons,
            'independently_parsed_query_streams':checked,'reviewed_files':reviewed,
            'neural_execution':False,'training':False,
            'limits':['Small exhaustive lifecycle alphabet: two keys/two values, four events; includes a third absent query key.',
                      'Public development fixture, only eight independent base blocks.',
                      'No runtime replay, scientific comparison, confidence interval or performance qualification.']}

if __name__=='__main__':
    result=run()
    print(json.dumps(result,indent=2,sort_keys=True))
