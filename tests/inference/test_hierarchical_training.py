import unittest
import torch
from relational_composition.model import Config
from relational_composition.hierarchical_model import PagedMultiHopLM
from relational_composition.train_hierarchical import hierarchical_credit_step


class HierarchicalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_active_page_bit_is_selected_and_params_remain_finite(self):
        m=PagedMultiHopLM(Config(width=8,layers=2,relations=2,addresses=2,candidates=2),page_bits=16)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        x=torch.randint(1,128,(2,48))
        y=x.roll(-1,1)
        answer=torch.zeros_like(y,dtype=torch.bool)
        answer[:,-5:]=True
        for seed in range(10):
            got=hierarchical_credit_step(m,opt,x,y,answer_mask=answer,
                    seed=seed,page_priority=1.,focus_prob=0.)
            self.assertEqual(got['interventions'],1)
            self.assertTrue(got['binary_conditional'])
            self.assertIn('page_',got['field'])
            self.assertEqual(got['forward_passes'],2)
            self.assertTrue(torch.isfinite(torch.tensor(got['gradient_norm'])))
        self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))

    def test_bad_arguments_are_rejected(self):
        m=PagedMultiHopLM(Config(width=8,layers=1),page_bits=8)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        x=torch.ones((1,4),dtype=torch.long)
        with self.assertRaises(ValueError):
            hierarchical_credit_step(m,opt,x,x,seed=1,page_priority=2)
        with self.assertRaises(ValueError):
            hierarchical_credit_step(m,opt,x,x,seed=1,answer_mask=torch.zeros((1,3),dtype=torch.bool))

if __name__=='__main__':
    unittest.main()
