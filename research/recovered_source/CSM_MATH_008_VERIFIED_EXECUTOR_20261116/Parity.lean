import TypedMemory
open TypedMemory
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.absent else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.value 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 0 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 1 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.absent else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.value 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 0 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 1 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.absent else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 0 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 1 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 0 else Cell.absent : TypedMemory.Memory Nat) 2 = some 0 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 0 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 1 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.value 1 else Cell.absent : TypedMemory.Memory Nat) 2 = some 1 := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 0 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 1 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 0 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 1 = none := by decide
example : TypedMemory.read 3 (fun i => if i = 0 then Cell.ref 2 else if i = 1 then Cell.ref 2 else if i = 2 then Cell.ref 2 else Cell.absent : TypedMemory.Memory Nat) 2 = none := by decide
