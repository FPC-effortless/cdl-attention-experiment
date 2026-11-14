import unittest,torch
from run import build,generate,validate,make_batch,assay_batch,weights_pin,final_set_interventions,categorical_only

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cases=generate()
    def models(self):
        for arm in ('csm','pointer_slot'):
            torch.manual_seed(5100);model=build(arm).eval();yield arm,model
    def test_grid_and_boundaries(self):
        self.assertEqual(validate(self.cases)['cases'],288)
        for c in self.cases:make_batch([c])
        with self.assertRaises(ValueError):make_batch(self.cases[:2])
        bad={**self.cases[0],'final_set_token':5}
        with self.assertRaises(ValueError):make_batch([bad])
        bad={**self.cases[0],'new':'v7'}
        with self.assertRaises(ValueError):make_batch([bad])
    def test_noop_equivalence_and_no_weight_change(self):
        with torch.inference_mode():
            for arm,m in self.models():
                for panel in ('baseline','overwrite'):
                    cs=[c for c in self.cases if c['panel']==panel][:4];b,_=make_batch(cs)
                    for heads in ('normal','categorical'):
                        before=weights_pin(m)
                        raw,_=m.sequence(b['ids'],supplied=categorical_only(b) if heads=='categorical' else None)
                        rec,traces,observed=assay_batch(m,cs,heads,False,False)
                        self.assertTrue(torch.equal(raw,torch.tensor([r['logits'] for r in rec])))
                        self.assertEqual(before,weights_pin(m));self.assertEqual(traces.shape,(4,3,32))
    def test_categorical_gain_not_overridden(self):
        cs=[c for c in self.cases if c['panel']=='overwrite'][:4];b,p=make_batch(cs)
        with torch.inference_mode():
            for arm,m in self.models():
                _,cap=m.sequence(b['ids'],collect=True,supplied=categorical_only(b))
                rec,_,observed=assay_batch(m,cs,'categorical',False,False)
                self.assertEqual([r['effective_final_gate'] for r in rec],cap['write_gate'][:,p].sigmoid().squeeze(-1).tolist())
                self.assertTrue(all(r['effective_final_edit']==1 for r in rec))
                self.assertTrue(all(r['effective_final_gate']!=1 for r in rec))
    def test_effective_full_gain_zero_reads_and_boundary(self):
        cs=[c for c in self.cases if c['panel']=='overwrite'][:4];b,p=make_batch(cs)
        with torch.inference_mode():
            for arm,m in self.models():
                x_before={};x_after={};gate_after={}
                h0=m.transform.register_forward_pre_hook(lambda mod,args:x_before.update({m.position:args[0].clone()}))
                try:
                    with final_set_interventions(m,p,zero_reads=True,full_gain=True) as calls:
                        h1=m.transform.register_forward_pre_hook(lambda mod,args:x_after.update({m.position:args[0].clone()}))
                        h2=m.write_gate.register_forward_hook(lambda mod,args,out:gate_after.update({m.position:out.clone()}))
                        try:m.sequence(b['ids'],supplied=categorical_only(b))
                        finally:h1.remove();h2.remove()
                    self.assertEqual(calls,{'zero_reads':1,'full_gain':1})
                    for step in range(27):
                        if step!=p:self.assertTrue(torch.equal(x_before[step],x_after[step]))
                        else:
                            self.assertTrue((x_after[step][:,32:96]==0).all())
                            self.assertTrue(torch.equal(x_before[step][:,:32],x_after[step][:,:32]))
                            self.assertTrue(torch.equal(x_before[step][:,96:],x_after[step][:,96:]))
                            self.assertTrue((gate_after[step]==32).all())
                    rec,_,observed=assay_batch(m,cs,'categorical',True,True)
                    self.assertTrue(all(r['effective_final_edit']==1 and r['effective_final_gate']==1 for r in rec))
                    self.assertTrue(all(r['query_preserved'] and r['untouched_set_preserved'] for r in rec))
                finally:h0.remove()
    def test_cleanup_missing_repeated_and_exception(self):
        cs=[c for c in self.cases if c['panel']=='baseline'][:2];b,p=make_batch(cs)
        with torch.inference_mode():
            for arm,m in self.models():
                counts=lambda:(len(m.transform._forward_pre_hooks),len(m.write_gate._forward_hooks))
                before=counts()
                with self.assertRaises(ValueError):
                    with final_set_interventions(m,p,zero_reads=True):pass
                self.assertEqual(before,counts())
                with self.assertRaises(ValueError):
                    with final_set_interventions(m,p,full_gain=True):
                        m.sequence(b['ids']);m.sequence(b['ids'])
                self.assertEqual(before,counts())
                with self.assertRaises(RuntimeError):
                    with final_set_interventions(m,p,zero_reads=True,full_gain=True):raise RuntimeError('test')
                self.assertEqual(before,counts())
    def test_invalid_dtype(self):
        for arm,m in self.models():
            with self.assertRaises(ValueError):assay_batch(m.double(),[self.cases[0]],'normal',False,False)

if __name__=='__main__':unittest.main()
