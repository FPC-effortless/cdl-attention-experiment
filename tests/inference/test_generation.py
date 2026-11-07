import unittest
import torch
from relational_composition.model import Config
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop
from relational_composition.generation import greedy_generate,grouped_greedy_generate


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(67)
        self.model=PagedMultiHopLM(Config(width=8,layers=2),page_bits=12).eval()

    def test_nonpad_greedy_stays_valid(self):
        with torch.no_grad():
            # Force PAD head to win BEFORE filtering. The API must suppress it.
            self.model.lm_head.weight.zero_()
            self.model.lm_head.bias.fill_(-20)
            self.model.lm_head.bias[0]=20
            self.model.lm_head.bias[3]=19
        x=torch.randint(1,100,(2,12))
        ret=greedy_generate(StreamingPagedMultiHop(self.model),x,max_new_tokens=40)
        self.assertTrue((ret.generated==3).all())
        self.assertEqual(ret.generated.shape,(2,40))
        self.assertEqual(ret.steps,40)
        self.assertTrue((ret.lengths==40).all())

    def test_eos_early_stop_and_zero_length_and_budget(self):
        with torch.no_grad():
            self.model.lm_head.weight.zero_()
            self.model.lm_head.bias.fill_(-10)
            self.model.lm_head.bias[8]=12
        x=torch.randint(1,100,(3,10))
        result=greedy_generate(StreamingPagedMultiHop(self.model),x,
                               max_new_tokens=40,eos_token_id=8)
        self.assertTrue((result.generated==8).all())
        self.assertTrue(result.terminated.all())
        self.assertEqual(result.steps,1)
        zero=greedy_generate(StreamingPagedMultiHop(self.model),x,max_new_tokens=0)
        self.assertEqual(zero.steps,0)
        with self.assertRaises(ValueError):
            greedy_generate(StreamingPagedMultiHop(self.model),x,
                    max_new_tokens=8,max_context_tokens=12)
        with self.assertRaises(ValueError):
            greedy_generate(StreamingPagedMultiHop(self.model),x,
                    max_new_tokens=8,eos_token_id=0)

    def test_grouped_ragged_length_isolation(self):
        prompts=[[8,9,10],[3,4,5,6],[15,16,17],[11]]
        results=grouped_greedy_generate(lambda:StreamingPagedMultiHop(self.model),
                       prompts,max_new_tokens=5,eos_token_id=8)
        self.assertEqual(len(results),len(prompts))
        for p,result in zip(prompts,results):
            single=greedy_generate(StreamingPagedMultiHop(self.model),
                        torch.tensor([p]),max_new_tokens=5,eos_token_id=8)
            self.assertEqual(result,single.generated[0,:int(single.lengths[0])].tolist())

if __name__=='__main__':
    unittest.main()
