import unittest
import torch
from resolved_state import ResolvedExactState
from data_adapter import ExactState

def payload(feature,index):
    raw=feature[0,index*22+3:index*22+6]
    return bytes(torch.round(raw*255).to(torch.uint8).tolist())
class ResolvedTests(unittest.TestCase):
    def test_all_records_not_only_query(self):
        s=ResolvedExactState();s.consume(b'The current code for Mira is ABC.\nKeep Sana\'s code linked to Mira.\nWhat code does Mira currently have?\nAnswer: ')
        f=s.feature();self.assertEqual(payload(f,0),b'ABC');self.assertEqual(payload(f,1),b'ABC')
        self.assertEqual(f[0,22+2],1);self.assertEqual(f[0,22+6],1)
        self.assertEqual(s.receipt()['resolved_records'],16)
    def test_absent_dangling_recreation(self):
        s=ResolvedExactState();s.consume(b"Keep Sana's code linked to Mira.\nWhat code does Sana currently have?\nAnswer: ")
        self.assertEqual(payload(s.feature(),0),b'\0\0\0')
        s.consume(b'The current code for Mira is ABC.\n')
        self.assertEqual(payload(s.feature(),0),b'ABC')
        s.consume(b'Remove the code for Mira.\n')
        f=s.feature();self.assertEqual(payload(f,0),b'\0\0\0');self.assertEqual(f[0,22],1)
        s.consume(b'The current code for Mira is DEF.\n')
        self.assertEqual(payload(s.feature(),0),b'DEF')
    def test_cycle_and_self_reference(self):
        s=ResolvedExactState();s.consume(b"Keep Sana's code linked to Mira.\nKeep Mira's code linked to Sana.\n")
        f=s.feature();self.assertEqual(payload(f,0),b'\0\0\0');self.assertEqual(payload(f,1),b'\0\0\0')
        self.assertLessEqual(s.receipt()['reference_resolution_hops'],32)
        s.consume(b"Keep Sana's code linked to Sana.\n")
        self.assertEqual(payload(s.feature(),0),b'\0\0\0')
    def test_raw_adapter_unchanged_and_snapshot(self):
        text=b"The current code for Mira is ABC.\nSave a snapshot of Mira's code for Sana.\nKeep Leah's code linked to Mira.\nThe current code for Mira is DEF.\n"
        raw=ExactState();new=ResolvedExactState();raw.consume(text);new.consume(text)
        self.assertEqual(payload(raw.feature(),2),b'\0\0\0')
        f=new.feature();self.assertEqual(payload(f,0),b'DEF');self.assertEqual(payload(f,1),b'ABC');self.assertEqual(payload(f,2),b'DEF')
        self.assertEqual(f.shape,(1,392));self.assertTrue(torch.equal(f[:,368:],torch.zeros(1,24)))
if __name__=='__main__':unittest.main()
