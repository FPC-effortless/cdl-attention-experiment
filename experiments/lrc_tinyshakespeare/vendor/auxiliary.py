"""Causal lexical training teachers, not semantic entity identity or runtime inputs."""
from bisect import bisect_left
import random
import torch
from torch.nn import functional as F

def teachers(x):
 # Writer h_j includes x_j, the continuation of prefix byte x_(j-1).
 # At t, retrieve only j<t for an identical previous prefix byte x_t.
 code=(x-1)%32;wr=code.roll(1,1);wr[:,0]=-100;gold=torch.full_like(x,-1)
 for row,values in enumerate(x.tolist()):
  latest={}
  for t,v in enumerate(values):
   gold[row,t]=latest.get(v,-1)
   if t>=1:latest[values[t-1]]=t
 return wr,code,gold

def selection_targets(out,gold,ell):
 labels=torch.zeros((*gold.shape,2),dtype=torch.long);valid=gold>=0
 for row in range(gold.shape[0]):
  rel=out.decisions[ell,'write_relation'][row].tolist();addr=out.decisions[ell,'write_address'][row].tolist();buckets={}
  for t,j in enumerate(gold[row].tolist()):
   if j>=0:
    bucket=buckets[(rel[j],addr[j])];rank=len(bucket)-1-bisect_left(bucket,j);labels[row,t,0]=rank//2;labels[row,t,1]=rank%2
   buckets.setdefault((rel[t],addr[t]),[]).append(t)
 return labels,valid

def auxiliary_loss(model,out,x,arm,seed):
 zero=out.logits.new_zeros(())
 if arm=='answer_only':return zero,{'address_aux':0.,'selection_aux':0.,'teacher_positions':0}
 wr,q,gold=teachers(x);b=x.shape[0];idx=torch.arange(b).roll(1) if arm=='shuffled_lexical_address_slot' else torch.arange(b);wr=wr[idx];q=q[idx];terms=[]
 wrrel=torch.where(wr>=0,wr//8,-100);wraddr=torch.where(wr>=0,wr%8,-100)
 for ell in range(2):
  terms.append(F.cross_entropy(out.policies[ell,'write_relation'].flatten(0,1),wrrel.flatten(),ignore_index=-100));terms.append(F.cross_entropy(out.policies[ell,'write_address'].flatten(0,1),wraddr.flatten(),ignore_index=-100))
  for hop,name in enumerate(['first','second']):
   terms.append(F.cross_entropy(out.policies[ell,'query_'+name].flatten(0,1),(q%8).flatten()));terms.append(F.cross_entropy(out.policies[ell,'routes'][:,:,hop].flatten(0,1),(q//8).flatten()))
 address=torch.stack(terms).mean();selection=zero;count=0
 if arm!='lexical_address':
  terms=[]
  for ell in range(2):
   labels,valid=selection_targets(out,gold,ell);labels=labels[idx];valid=valid[idx];count+=int(valid.sum())
   if not valid.any():continue
   bits=(labels[...,0,None]>>torch.arange(8))&1
   for name in ['first','second']:
    terms.append(F.cross_entropy(out.policies[ell,'rank_'+name][valid],labels[...,1][valid]));terms.append(F.cross_entropy(out.policies[ell,'page_'+name][valid].flatten(0,1),bits[valid].flatten()))
  if terms:selection=torch.stack(terms).mean()
 return .1*(address+selection),{'address_aux':float(address.detach()),'selection_aux':float(selection.detach()),'teacher_positions':count}
