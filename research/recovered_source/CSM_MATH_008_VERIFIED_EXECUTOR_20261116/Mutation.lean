import TypedMemory
open TypedMemory
def badCopy (m : Memory Nat) (i j : Nat) := link m i j
example : read 3 (set (badCopy (set empty 0 7) 1 0) 0 9) 1 = some 7 := by decide
