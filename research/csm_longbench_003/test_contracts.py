import torch
from experiment import corpus,panels,model,evaluate,detach

def main():
 torch.set_num_threads(1);torch.manual_seed(81)
 train,dev=corpus();panel,starts=panels(dev)
 assert len(panel)==16 and panel.shape[1]==2049
 assert starts[-1]+2049==len(dev)
 m=model('instant');x=panel[:2,:65]
 a=evaluate(m,x,64);b=evaluate(m,x,64,'trace_reset');c=evaluate(m,x,64,'trace_swap')
 assert a['nll']==b['nll']==c['nll']
 m=model('multi');s=m.init_state(2,'cpu');ys=[]
 for t in range(64):
  y,s,_=m.step(x[:,t],s);ys.append(y.detach())
 s=m.init_state(2,'cpu');chunk=[]
 for t in range(64):
  if t==32:s=detach(s)
  y,s,_=m.step(x[:,t],s);chunk.append(y.detach())
 assert torch.equal(torch.stack(ys),torch.stack(chunk))
 print('PASS: split windows, immediate reset/swap controls, detach preserves forward')
if __name__=='__main__':main()
