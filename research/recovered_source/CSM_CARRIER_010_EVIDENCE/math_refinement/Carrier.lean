import TypedMemory
open TypedMemory
namespace Carrier
-- Code zero is absent; positive k encodes payload k-1.
def decode (k : Nat) : Cell Nat := if k = 0 then .absent else .value (k-1)
def advance (c a : Nat) := if a > 0 then a else c
def execute (c a : Nat) : Cell Nat := if a = 0 then decode c else .value (a-1)
theorem update_refinement (c a : Nat) : decode (advance c a) = execute c a := by
 cases a with
 | zero => simp [advance, execute]
 | succ k => simp [advance, execute, decode]
theorem no_action_preserves (c : Nat) : decode (advance c 0) = decode c := by simp [advance]
theorem action_overwrites (c k : Nat) : decode (advance c (k+1)) = .value k := by simp [advance, decode]
#print axioms update_refinement
#print axioms no_action_preserves
#print axioms action_overwrites
end Carrier
