"""Exhaustive finite predicate input diagnostic, NOT end-to-end learned memory."""
import torch
from torch.nn import functional as F
def probe(model):
 rows=[]
 with torch.inference_mode():
  for source in range(5):
   for expected in range(1,5):
    target=int(source==expected)
    if model.arm=='local_exact':prediction=target;kind='supplied exact comparator'
    else:
     x=torch.cat((F.one_hot(torch.tensor([source]),5),F.one_hot(torch.tensor([expected-1]),4)),-1).float()
     prediction=int(model.gate_network(x).argmax(-1)[0]);kind='learned gate only; supplied operand representation'
    rows.append(dict(source=source,expected=expected,target=target,prediction=prediction))
 return dict(kind=kind,correct=sum(r['prediction']==r['target'] for r in rows),total=20,rows=rows,scope='Finite isolated predicate table; no parser/routing/state-retention or general-language claim')
