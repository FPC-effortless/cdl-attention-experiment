import unittest,torch,json
from model_source import build,tensors,objective,history_map,VOCAB
from paired_data import generate,validate,example
from learn import forward_with_root,loss_parts
from experiment import make_bank,batch_at,schedule,replay

class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.rows=generate();cls.pairs=[r['observation']['pair'] for r in cls.rows]
    def test_semantics_unique_and_pair_alignment(self):
        self.assertEqual(validate(self.rows,replay)['distinct_histories'],1600)
        for p in self.pairs:
            a,b=p['worlds'];ma,mb=history_map(a),history_map(b)
            if p['focal_op']=='copy' and p['template']!='detach_alias' and p['template']!='delete_alias':
                self.assertNotEqual(ma[p['alias']],mb[p['alias']])
            if p['template']=='absence_snapshot':
                self.assertIsNone(ma[p['root']]);self.assertIsNone(ma[p['spare']])
    def test_bank_pair_query_and_schedule(self):
        bank=make_bank(self.pairs[:10]);i=torch.tensor([2,7]);q=torch.tensor([0,3]);b,r=batch_at(bank,self.pairs[:10],i,q)
        for j,(pi,qi) in enumerate(zip(i.tolist(),q.tolist())):
            for wi in range(2):
                expected=example(self.pairs[pi]['worlds'][wi],f'r{qi}')
                for k in b:self.assertEqual(b[k][2*j+wi].tolist(),expected[k])
        self.assertEqual(r.tolist(),[int(self.pairs[pi]['root'][1:]) for pi in i.tolist() for _ in range(2)])
        a,c=schedule(101,640);x,y=schedule(101,640);self.assertTrue(torch.equal(a,x) and torch.equal(c,y))
    def test_capture_forward_and_zero_weight_gradients(self):
        bank=make_bank(self.pairs[:10]);b,roots=batch_at(bank,self.pairs[:10],torch.tensor([0,1]),torch.tensor([0,1]))
        for arm in ('csm','pointer_slot'):
            torch.manual_seed(52);m=build(arm)
            logits,cap=m.sequence(b['ids'],collect=True);loss,_=objective(logits,cap,b);loss.backward()
            old={k:p.grad.clone() if p.grad is not None else None for k,p in m.named_parameters()};m.zero_grad(set_to_none=True)
            new,cap,state=forward_with_root(m,b,roots);loss2,_=loss_parts(new,cap,state,b,0)
            self.assertTrue(torch.equal(logits,new));self.assertTrue(torch.equal(loss,loss2));self.assertTrue(state.requires_grad)
            loss2.backward()
            for k,p in m.named_parameters():
                if old[k] is not None:self.assertTrue(torch.allclose(old[k],p.grad,atol=1e-7,rtol=1e-6),k)
    def test_invariance_gradient_and_capture_boundary(self):
        bank=make_bank(self.pairs[:10]);b,roots=batch_at(bank,self.pairs[:10],torch.tensor([0,1]),torch.tensor([0,1]))
        for arm in ('csm','pointer_slot'):
            torch.manual_seed(53);m=build(arm);_,_,state=forward_with_root(m,b,roots)
            penalty=(state[0::2]-state[1::2]).square().mean();penalty.backward()
            self.assertGreater(float(penalty),0)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None),0)
            # Prefix execution reproduces the capture at exactly the final SET.
            with torch.no_grad():
                s=m.init_state(4,'cpu')
                for pos,token in enumerate(b['ids'][:,:18].unbind(1)):
                    m.position=pos;_,s=m.step(token,s)
                self.assertTrue(torch.equal(state.detach(),s.values[torch.arange(4),roots]))
    def test_loss_uses_only_pair_root_states(self):
        # Direct loss derivative is confined to the supplied root vectors; query logits remain independent arguments.
        states=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]],requires_grad=True)
        p=(states[0::2]-states[1::2]).square().mean();p.backward()
        self.assertTrue(torch.equal(states.grad,torch.tensor([[-1.,-1.],[1.,1.],[-1.,-1.],[1.,1.]])))

if __name__=='__main__':unittest.main()
