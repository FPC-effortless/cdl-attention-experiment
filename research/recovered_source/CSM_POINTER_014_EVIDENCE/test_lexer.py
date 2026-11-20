"""Zero-fit raw-byte lexer checks, including actual prospective 014 histories."""
import json, unittest
from pathlib import Path
import torch
from lexer import init,scan

def raw(s):return torch.tensor([list(s.encode('ascii'))],dtype=torch.long)
class LexerTests(unittest.TestCase):
 def test_boundaries(self):
  text='Set A to a. Ignore: Copy B from A. XX C1 D.'
  r,l,c,st,tr=scan(raw(text),init(1),True)
  # XX is not a one-letter token; numeric neighbor is nonletter by contract.
  self.assertEqual(st['nextslot'].item(),4)
  self.assertEqual([i for i in range(26) if st['table'][0,i]>=0],[0,1,2,3])
  self.assertEqual(r[0,0].item(),257)
  self.assertEqual(r[0,1].item(),ord('S'))
  self.assertTrue(all(l[0,i]==256 for i,ch in enumerate(text) if ch.isupper()))
 def test_pending(self):
  a=scan(raw('A'),init(1));self.assertEqual(a[3]['nextslot'].item(),0)
  b=scan(raw('b '),a[3]);self.assertEqual(b[3]['nextslot'].item(),0)
  d=scan(raw(' '),a[3]);self.assertEqual(d[3]['nextslot'].item(),1)
 def test_atomic_capacity(self):
  st=scan(raw('A B C D '),init(1))[3];before={k:v.clone() for k,v in st.items()}
  with self.assertRaisesRegex(ValueError,'fifth'):scan(raw('E '),st)
  for k in st:self.assertTrue(torch.equal(st[k],before[k]))
 def test_future_noninterference(self):
  a=scan(raw('Set A to a. Read A.'),init(1))
  b=scan(raw('Set A to a. Read Z.'),init(1))
  # The differing identity byte is future relative to the retained prefix.
  for idx in (0,1,2,4):self.assertTrue(torch.equal(a[idx][:,:15],b[idx][:,:15]))
 def test_full_split_raw_panels(self):
  d=json.loads((Path(__file__).resolve().parents[1]/'data.json').read_text())
  for panels in d.values():
   for rows in panels.values():
    # One representative from every raw condition; all split points include clauses.
    x=raw(rows[0]['text']);whole=scan(x,init(1))
    for cut in [1,17,x.shape[1]//2,x.shape[1]-1]:
     a=scan(x[:,:cut],init(1));b=scan(x[:,cut:],a[3])
     for idx in [0,1,2,4]:self.assertTrue(torch.equal(whole[idx],torch.cat([a[idx],b[idx]],1)))
     for k in whole[3]:self.assertTrue(torch.equal(whole[3][k],b[3][k]))
 def test_renaming(self):
  d=json.loads((Path(__file__).resolve().parents[1]/'data.json').read_text())['dev']
  for key,rows in d.items():
   if key.endswith('_renamed_EH'):continue
   others=d[key+'_renamed_EH']
   for a,b in zip(rows,others):
    out=scan(raw(a['text']),init(1));ren=scan(raw(b['text']),init(1))
    for idx in (0,1,2):self.assertTrue(torch.equal(out[idx],ren[idx]))
    # Untied mode retains resolved raw uppercase identities, not role labels.
   u=scan(raw(rows[0]['text']),init(1),False);v=scan(raw(others[0]['text']),init(1),False)
   self.assertFalse(torch.equal(u[0],v[0]));self.assertTrue(torch.equal(u[2],v[2]))
if __name__=='__main__':
 torch.set_num_threads(1);unittest.main()
