import copy
import unittest
import torch
from data import generate,examples,encode,history_map,HEADS
from models import CSM,PointerSlot,tensors,objective
from native_csm import RecordRewrite,Config
from investigation_world.benchmark.lifecycle_observability import replay

class ResearchChecks(unittest.TestCase):
    def test_all_generated_labels_match_independent_veritas(self):
        rows=generate();self.assertEqual(len(rows),1280)
        for row in rows:
            mem=row['observation']['memory']
            self.assertEqual(history_map(mem),replay(mem))
            if row['observation']['condition']['template']=='recreate_snapshot':
                changed=copy.deepcopy(mem)
                changed['events']=[{'op':'nop'} if e['op']=='delete' else e for e in mem['events']]
                role=row['observation']['condition']['roles']['snapshot']
                self.assertIsNone(history_map(mem)[role])
                self.assertIsNotNone(history_map(changed)[role])

    def test_role_targets_and_masks(self):
        for row in generate()[:40]:
            mem=row['observation']['memory']
            for q in mem['keys']:
                x=encode(mem,q)
                self.assertEqual(len(x['ids']),27)
                for i,e in enumerate(mem['events']):
                    p=i*3+2;op=e['op'];dest=int(e['key'][1:])
                    self.assertEqual(x['destination'][p],dest)
                    self.assertEqual(x['second_address'][p],int(e['source'][1:]) if op=='link' else dest)
                    self.assertEqual(x['value_edit'][p],int(op!='link'))
                    self.assertEqual(x['use_link'][p],0)
                for p in range(27):
                    if p%3!=2:
                        self.assertEqual(x['value_edit'][p],0)
                        self.assertEqual(x['link_edit'][p],0)
                        self.assertEqual(x['destination'][p],-100)
                self.assertEqual(x['destination'][-1],7)
                self.assertEqual(x['second_address'][-1],-100)
                self.assertEqual(x['first_address'][-1],int(q[1:]))
                self.assertEqual(x['link_edit'][-1],0)
                self.assertEqual(x['use_link'][-1],1)

    def test_native_parity_no_forward_rewrite(self):
        torch.manual_seed(90);model=CSM();reference=RecordRewrite(Config(vocab_size=20))
        reference.load_state_dict(model.state_dict());ids=torch.randint(20,(3,27))
        self.assertTrue(torch.equal(model.sequence(ids)[0],reference(ids)[:,-1]))
        self.assertTrue(torch.equal(model.sequence(ids,collect=True)[0],reference(ids)[:,-1]))

    def test_equal_head_information_and_gradients(self):
        row=generate()[0]
        forged={'record_id':row['record_id'],'subject_id':row['subject_id'],'features':row['observation']}
        bank=tensors(examples([forged]));counts=[]
        for cls in (CSM,PointerSlot):
            model=cls();counts.append(sum(p.numel() for p in model.parameters()))
            logits,cap=model.sequence(bank['ids'],collect=True)
            self.assertEqual(set(cap),set(HEADS)|{'write_gate'})
            loss,_=objective(logits,cap,bank);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertIsNone(model.supplied);self.assertIsNone(model.capture)
        self.assertEqual(counts,[18359,18357])

    def test_literal_duplicate_histories_share_subject(self):
        import json
        seen={}
        for row in generate():
            signature=json.dumps(row['observation']['memory']['events'],sort_keys=True)
            subject=row['subject_id']
            self.assertEqual(seen.setdefault(signature,subject),subject)

    def test_exact_semantic_falsifiers(self):
        common=[{'op':'set','key':'r0','value':'v0'},{'op':'set','key':'r3','value':'v2'}]
        for focal in ('copy','link'):
            m={'schema':'woyengi.lifecycle.v1','keys':['r0','r1','r2','r3'],'composed_keys':['r1','r2'],
               'events':common+[{'op':focal,'key':'r1','source':'r0'},{'op':'delete','key':'r0'},
                   {'op':'copy','key':'r2','source':'r0'},{'op':'set','key':'r0','value':'v1'}]}
            self.assertEqual(history_map(m),{'r0':'v1','r1':'v0' if focal=='copy' else 'v1','r2':None,'r3':'v2'})

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
