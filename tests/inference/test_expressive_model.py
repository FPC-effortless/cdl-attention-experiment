"""Architectural capability and cached-vs-bulk parity, not learned competence."""
import unittest
import torch
from relational_composition.model import Config
from relational_composition.expressive_model import SerialMultiHopLM, StreamingMultiHop

class ExpressivityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_two_distinct_typed_reads_and_dependent_second_policy(self):
        torch.manual_seed(29)
        model = SerialMultiHopLM(Config(width=10, layers=1, candidates=3)).eval()
        x = torch.randint(1,130,(1,18))
        base = model(x)
        # Force all writes to one namespace/address; route 1 uses r=0, route 2 r=1.
        forced = {k: v.clone() for k,v in base.decisions.items()}
        forced[(0,'write_relation')].fill_(0)
        forced[(0,'write_address')].fill_(0)
        forced[(0,'routes')][:,:,0] = 0
        forced[(0,'routes')][:,:,1] = 1
        forced[(0,'query_first')].fill_(0)
        forced[(0,'rank_first')].fill_(0)
        first = model(x, overrides=forced)
        self.assertTrue((first.operands[0][:,:,1] == -1).all())  # namespace 1 empty
        self.assertTrue((first.operands[0][0,1:,0] >= 0).all())
        alternative = {k:v.clone() for k,v in forced.items()}
        alternative[(0,'rank_first')][:,3:] = 1
        changed = model(x, overrides=alternative)
        self.assertFalse(torch.equal(first.policies[(0,'query_second')][:,3:],
                                     changed.policies[(0,'query_second')][:,3:]))
        # Strictly causal even with distinct namespaces and dependent hops.
        for ids in changed.operands:
            self.assertTrue((ids < torch.arange(18)[None,:,None]).all())

    def test_streaming_parity_and_prefill_handoff(self):
        for variant in ('coupled','no_reference'):
            for seed in (4,11):
                torch.manual_seed(seed)
                model = SerialMultiHopLM(Config(width=12, layers=3, variant=variant)).eval()
                x = torch.randint(1,100,(2,24))
                fast = StreamingMultiHop(model)
                for t in range(1,13):
                    last = fast.step(x[:,t-1])
                    bulk = model(x[:,:t])
                    torch.testing.assert_close(last,bulk.logits[:,-1],atol=2e-5,rtol=2e-5)
                fast.prefill(x[:,:17])
                for t in range(18,25):
                    last = fast.step(x[:,t-1])
                    bulk = model(x[:,:t])
                    torch.testing.assert_close(last,bulk.logits[:,-1],atol=2e-5,rtol=2e-5)
                self.assertEqual(fast.cache.count,bulk.metrics)

    def test_model_training_graph_is_differentiable_for_selected_values(self):
        torch.manual_seed(5)
        model = SerialMultiHopLM(Config(width=8,layers=2))
        x = torch.randint(1,130,(1,13))
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.logits.reshape(-1,257),x.flatten())
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in model.parameters()))

if __name__=='__main__':
    unittest.main()
