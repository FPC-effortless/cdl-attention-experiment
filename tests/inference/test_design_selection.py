import unittest
from relational_composition.design_selection import Candidate, select_qualified

class SelectionTests(unittest.TestCase):
    def test_competence_beats_impressive_latency(self):
        a=Candidate('fast-but-no-multihop',2.,100.,.0,True,False,15.)
        b=Candidate('capable',5.,400.,.95,True,True,40.)
        selection=select_qualified([a,b],reference_inference_ms=10,
                                    reference_training_ms=1000)
        self.assertEqual(selection['winner'],'capable')
        selection=select_qualified([a],reference_inference_ms=10,
                                    reference_training_ms=1000)
        self.assertIsNone(selection['winner'])
    def test_inference_priority_after_gates(self):
        a=Candidate('fast-inference',5.,1000.,.95,True,True,40.)
        b=Candidate('fast-training',10.,100.,.95,True,True,40.)
        s=select_qualified([a,b],reference_inference_ms=10,
                            reference_training_ms=1000)
        self.assertEqual(s['winner'],'fast-inference')

if __name__=='__main__':
    unittest.main()
