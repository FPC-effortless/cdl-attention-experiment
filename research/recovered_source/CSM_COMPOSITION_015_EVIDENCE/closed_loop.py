"""Oracle intervention at action boundary, then regenerate from corrected owned memory."""
import torch
from model import Model
import lexer,executor
MODES=['original','both_identities','positive_actions','suppress_false','all_gold_ceiling']
def canonical(a):
 op,dst,src,val=a.unbind(-1)
 return torch.stack([op,torch.where(op>0,dst,0),torch.where((op==2)|(op==3),src,0),torch.where(op==1,val,0)],-1)
def rollout(m,x,state,gold,mode):
 mem,hist,lexstate=state;resolved,current,candidates,lexnext,_=lexer.scan(x,lexstate,tied=m.tied);raw=m.embedding(resolved);look=m.embedding(current)
 if m.is_gru:front,newhist=m.gru(torch.cat((raw,look),-1),hist)
 else:
  context=torch.cat((hist,raw),1);windows=context.unfold(1,48,1).permute(0,1,3,2).flatten(2);front=m.frontend(torch.cat((windows,look),-1)).tanh();newhist=context[:,-47:]
 if not m.conditioned:logits=m.decode(m.head(front),candidates)
 else:features=executor.features(mem);ls=[]
 answers=[];ks=[];ds=[];raws=[];executed=[]
 for t in range(x.shape[1]):
  if m.conditioned:l=m.decode(m.head(torch.cat((front[:,t],features),-1)),candidates[:,t]);ls.append(l)
  else:l=logits[:,t]
  op,dst,src,val=(z.argmax(-1) for z in l.split([6,4,4,4],-1));a=canonical(torch.stack((op,dst,src,val+1),-1));b=a.clone();g=gold[:,t]
  if mode=='both_identities':
   mask=((g[:,0]==2)|(g[:,0]==3))&(a[:,0]==g[:,0]);b[:,1]=torch.where(mask,g[:,1],b[:,1]);b[:,2]=torch.where(mask,g[:,2],b[:,2])
  elif mode=='positive_actions':b=torch.where((g[:,0]>0)[:,None],g,b)
  elif mode=='suppress_false':b=torch.where((g[:,0]==0)[:,None],torch.zeros_like(b),b)
  elif mode=='all_gold_ceiling':b=g.clone()
  elif mode!='original':raise ValueError(mode)
  op,dst,src,val=b.unbind(-1)
  if bool((op!=0).any()):
   mem,ans=executor.step(mem,op,dst,src,val)
   if m.conditioned and bool(((op>=1)&(op<=4)).any()):features=executor.features(mem)
  else:ans=torch.full_like(op,-1)
  raws.append(a);executed.append(b);answers.append(ans);ks.append(mem[0]);ds.append(mem[1])
 if m.conditioned:logits=torch.stack(ls,1)
 return logits,torch.stack(answers,1),(mem,newhist,lexnext),(torch.stack(ks,1),torch.stack(ds,1)),torch.stack(raws,1),torch.stack(executed,1)

def gold(cases):
 y=torch.zeros(len(cases),len(cases[0]['text']),4,dtype=torch.long)
 for i,q in enumerate(cases):
  for e in q['events']:y[i,e['pos']]=torch.tensor([e['op'],e['dst'],e['src'],e['value']])
 return y

def nonnoop(a):return [[i,*v] for i,v in enumerate(a.tolist()) if v[0]>0]
def record(q,result,index,arm,mode,baseline):
 logits,ans,st,tr,raw,act=result;kind,data=tr;pos=q['query_positions'];fpos=q['final_query_positions'];g={e['pos']:e['answer'] for e in q['events'] if e['op']==5}
 output=dict(case_id=q['case_id'],family=q['family'],arm=arm,length=len(q['text']),pred_nonnoop=nonnoop(act[index]),gold_nonnoop=[[e['pos'],e['op'],e['dst'],e['src'],e['value']] for e in q['events'] if e['op']],pred_emitted_query_answers=[[t,int(ans[index,t])] for t in range(ans.shape[1]) if act[index,t,0]==5],query_positions=pos,query_targets=[g[p] for p in pos],query_predictions=[int(ans[index,p]) for p in pos],final_query_positions=fpos,final_query_targets=[g[p] for p in fpos],final_query_predictions=[int(ans[index,p]) for p in fpos],terminal_positions=[e['pos'] for e in q['events']],gold_terminal_states=[e['state_after'] for e in q['events']],pred_terminal_states=[[[int(kind[index,e['pos'],j]),int(data[index,e['pos'],j])] for j in range(4)] for e in q['events']],gold_final_state=q['final_state'],pred_final_state=[[int(st[0][0][index,j]),int(st[0][1][index,j])] for j in range(4)])
 output.update(mode=mode,raw_nonnoop=nonnoop(raw[index]),forced_positions=int((raw[index]!=act[index]).any(-1).sum()),raw_decision_changes=int((raw[index]!=baseline[4][index]).any(-1).sum()))
 return output
