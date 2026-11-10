"""A saved/restored optimizer step must equal an uninterrupted step."""
import io
import torch
from experiment import model,corpus,detach,State,base,state_dict

def step(m,o,s,x):
 o.zero_grad();ys=[]
 for t in range(16):
  y,s,_=m.step(x[:,t],s);ys.append(y)
 loss=torch.nn.functional.cross_entropy(torch.stack(ys,1).flatten(0,1),x[:,1:17].flatten())
 loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
 return detach(s)
def main():
 torch.set_num_threads(1);torch.manual_seed(90);data,_=corpus();x=data[:34].reshape(2,17)
 m=model('single');o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);s=step(m,o,m.init_state(2,'cpu'),x)
 buf=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),state=state_dict(s),rng=torch.get_rng_state()),buf)
 original=step(m,o,s,x)
 buf.seek(0);ck=torch.load(buf,weights_only=True);n=model('single');no=torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.01)
 n.load_state_dict(ck['model']);no.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);a=ck['state']
 ns=State(base.RecordState(a['values'],a['links'],a['cursor'],a['history'],a['position']),a['regulation']);ns=step(n,no,ns,x)
 assert all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items())
 assert torch.equal(original.core.values,ns.core.values) and torch.equal(original.regulation,ns.regulation)
 for k,v in o.state_dict()['state'].items():
  for name,t in v.items():assert torch.equal(t,no.state_dict()['state'][k][name])
 print('PASS: exact restored model, AdamW state, and recurrent state after next update')
if __name__=='__main__':main()
