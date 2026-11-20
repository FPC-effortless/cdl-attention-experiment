"""External supervision/scoring conversion; never called by model forward."""
import copy,torch
import lexer

def convert(cases):
 output=[]
 x=torch.tensor([list(case['text'].encode()) for case in cases]);_,_,_,last,alltables=lexer.scan(x,lexer.init(len(cases)));alltables=alltables.tolist()
 for row,case in enumerate(cases):
  symbols=case['identity_symbols'];ids=[ord(c)-65 for c in symbols];tables=alltables[row];q=copy.deepcopy(case)
  q['label_space']='first-occurrence-interned-slots'
  q['program_label_space']='logical-identity-indices'
  q['logical_events']=copy.deepcopy(case['events'])
  q['logical_final_state']=copy.deepcopy(case['final_state'])
  q['canonical_table_final']=last['table'][row].tolist()
  def mapped_state(st,table):
   out=[[0,0] for _ in range(4)]
   for i,(k,v) in enumerate(st):
    dst=table[ids[i]]
    if dst<0:
     assert k==0;continue
    out[dst]=[k,table[ids[v]] if k==2 else v]
    if k==2:assert out[dst][1]>=0
   return out
  for e in q['events']:
   tab=tables[e['pos']]
   if e['op']>0:e['dst']=tab[ids[e['dst']]];assert e['dst']>=0
   if e['op'] in (2,3):e['src']=tab[ids[e['src']]];assert e['src']>=0
   e['state_after']=mapped_state(e['state_after'],tab)
  q['final_state']=mapped_state(q['final_state'],last['table'][row].tolist());output.append(q)
 return output
