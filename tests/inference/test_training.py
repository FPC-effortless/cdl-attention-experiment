import unittest
import torch
from relational_composition.model import Config
from relational_composition.expressive_model import SerialMultiHopLM
from relational_composition.train_targeted import targeted_counterfactual_step

class TrainingStepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_targeted_credit_includes_writes_and_dependent_second_query(self):
        torch.manual_seed(42)
        model=SerialMultiHopLM(Config(width=8,layers=2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randint(1,120,(2,20)); y=x.roll(-1,1)
        mask=torch.zeros_like(x,dtype=torch.bool);mask[:,17:19]=True
        for key in ((0,'write_address'),(0,'query_second')):
            trace=targeted_counterfactual_step(model,optimizer,x,y,answer_mask=mask,
                         seed=31,decision_key=key,focus_prob=1.)
            self.assertEqual(trace['interventions'],1)
            self.assertTrue(trace['focused'])
            self.assertEqual(trace['full_forward_passes'],2)
            self.assertTrue(torch.isfinite(torch.tensor(trace['nll'])))

if __name__ == '__main__':
    unittest.main()
