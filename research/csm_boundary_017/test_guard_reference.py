import unittest
from guard import select_operation
class GuardTests(unittest.TestCase):
 def test_every_byte(self):
  for byte in range(256):
   for action in range(6):
    scores=[-1e300]*6;scores[action]=1e300
    self.assertEqual(select_operation(scores,byte),action if byte in (46,63) else 0)
 def test_no_forced_write(self):
  for byte in (46,63):self.assertEqual(select_operation([1,0,0,0,0,0],byte),0)
 def test_unguarded(self):self.assertEqual(select_operation([0,1,0,0,0,0],65,False),1)
 def test_nonfinite_rejected(self):
  for v in [float('inf'),-float('inf'),float('nan')]:
   with self.assertRaises(ValueError):select_operation([v,0,0,0,0,0],65)
if __name__=='__main__':unittest.main()
