import unittest
import torch
from relational_composition.model import Config
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop


class HierarchicalModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cached_bulk_parity_for_three_seeds_and_multiple_hops(self):
        for seed in (11,25,37):
            for variant in ('coupled','no_reference'):
                torch.manual_seed(seed)
                cfg=Config(width=10,layers=2,candidates=2,local_kernel=5,variant=variant)
                model=PagedMultiHopLM(cfg,page_bits=14).eval()
                x=torch.randint(1,128,(2,29))
                runner=StreamingPagedMultiHop(model)
                for t in range(1,12):
                    actual=runner.step(x[:,t-1])
                    expected=model(x[:,:t])
                    torch.testing.assert_close(actual,expected.logits[:,-1],atol=2e-5,rtol=2e-5)
                    self.assertEqual(runner.cache.count,expected.metrics)
                runner.prefill(x[:,:19])
                for t in range(20,30):
                    actual=runner.step(x[:,t-1])
                    expected=model(x[:,:t])
                    torch.testing.assert_close(actual,expected.logits[:,-1],atol=2e-5,rtol=2e-5)
                    self.assertEqual(runner.cache.count,expected.metrics)
                    for positions in expected.operands:
                        self.assertTrue((positions < torch.arange(t)[None,:,None]).all())

    def test_older_page_is_usable_with_forced_code(self):
        torch.manual_seed(42)
        cfg=Config(width=8,layers=1,candidates=2)
        model=PagedMultiHopLM(cfg,page_bits=8).eval()
        x=torch.randint(1,100,(1,12))
        first=model(x)
        choices={k:v.clone() for k,v in first.decisions.items()}
        choices[(0,'write_relation')].fill_(0)
        choices[(0,'write_address')].fill_(0)
        choices[(0,'routes')].fill_(0)
        choices[(0,'query_first')].fill_(0)
        choices[(0,'query_second')].fill_(0)
        choices[(0,'rank_first')].fill_(0)
        choices[(0,'rank_second')].fill_(0)
        choices[(0,'page_first')].fill_(0)
        choices[(0,'page_second')].fill_(0)
        newest=model(x,overrides=choices)
        self.assertEqual(int(newest.operands[0][0,-1,0]),10)
        older={k:v.clone() for k,v in choices.items()}
        older[(0,'page_first')][0,-1,0]=1
        old=model(x,overrides=older)
        self.assertEqual(int(old.operands[0][0,-1,0]),8)
        self.assertNotEqual(int(newest.operands[0][0,-1,0]),int(old.operands[0][0,-1,0]))
        self.assertTrue(old.active_page_bits[(0,'page_first')][0,-1,0])
        self.assertFalse(old.active_page_bits[(0,'page_first')][0,-1,7])

    def test_gumbel_replay_gradient_and_padding(self):
        torch.manual_seed(10)
        model=PagedMultiHopLM(Config(width=9,layers=2),page_bits=12)
        x=torch.randint(1,130,(2,17))
        x[1,12:]=0
        sampled=model(x,sample=True)
        repeated=model(x,sample=True,noise=sampled.noise)
        torch.testing.assert_close(sampled.logits,repeated.logits,atol=0,rtol=0)
        for k in sampled.decisions:
            self.assertTrue(torch.equal(sampled.decisions[k],repeated.decisions[k]))
        y=x.clone()
        loss=torch.nn.functional.cross_entropy(sampled.logits.flatten(0,1),y.flatten(),ignore_index=0)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

if __name__=='__main__':
    unittest.main()
