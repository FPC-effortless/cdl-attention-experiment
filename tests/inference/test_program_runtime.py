import unittest
import numpy as np
from relational_composition.program_runtime import VersionedProgramEngine, serial_batch


class ProgramEngineTests(unittest.TestCase):
    def test_equivalence_and_update_invalidation(self):
        rng=np.random.default_rng(17)
        tables={r:rng.permutation(193) for r in range(5)}
        engine=VersionedProgramEngine(tables)
        paths=([1],[0,2],[1,1,2,3,0]*16,[],[2,0,4,1]*64)
        starts=rng.integers(0,193,size=1024)
        for path in paths:
            baseline=serial_batch(starts,path,engine.tables)
            for mode in ('serial','compiled','auto'):
                got,info=engine.execute(starts,path,mode=mode)
                np.testing.assert_array_equal(got,baseline)
            _,info=engine.execute(starts,path,mode='compiled')
            self.assertTrue(info.compiled_cache_hit)
        prior=engine.compile_count
        engine.update(1,7,190)
        self.assertEqual(engine.epoch,1)
        got,info=engine.execute(starts,paths[2],mode='compiled')
        self.assertFalse(info.compiled_cache_hit)
        np.testing.assert_array_equal(got,serial_batch(starts,paths[2],engine.tables))
        self.assertEqual(engine.compile_count,prior+1)

    def test_auto_batched_vs_cheap_single(self):
        rng=np.random.default_rng(99)
        engine=VersionedProgramEngine({r:rng.permutation(1000) for r in range(3)})
        path=[0,1,2]*32
        _,a=engine.execute([7],path,mode='auto')
        self.assertEqual(a.mode,'serial')
        _,b=engine.execute(np.arange(1000),path,mode='auto',anticipated_queries=100000)
        self.assertEqual(b.mode,'compiled')
        with self.assertRaises(ValueError):
            engine.update(9,0,0)

if __name__=='__main__':
    unittest.main()

# More focused property regression: externally held table snapshots cannot
# silently modify cached plans, and unrelated updates preserve valid programs.
class ProgramInvalidationTests(unittest.TestCase):
    def test_update_dependency_and_safe_public_snapshots(self):
        engine=VersionedProgramEngine({0:np.array([1,2,0]),1:np.array([2,0,1])})
        before,info=engine.execute([0,1,2],[0,0],mode='compiled')
        snapshot=engine.tables
        with self.assertRaises(ValueError):
            snapshot[0][0]=2
        with self.assertRaises(TypeError):
            snapshot[2]=np.array([0,1,2])
        engine.update(1,0,1)  # unrelated relation, compiled [0,0] remains correct
        after,info=engine.execute([0,1,2],[0,0],mode='compiled')
        np.testing.assert_array_equal(after,before)
        self.assertTrue(info.compiled_cache_hit)
        # A dependency update correctly invalidates [0,0].
        engine.update(0,0,2)
        after2,info=engine.execute([0,1,2],[0,0],mode='compiled')
        self.assertFalse(info.compiled_cache_hit)
        np.testing.assert_array_equal(after2,serial_batch([0,1,2],[0,0],engine.tables))
