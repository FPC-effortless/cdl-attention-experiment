import Std
namespace Carry
def step (c a : Nat) : Nat := if a = 0 then c else a
def latest : List Nat → Option Nat
 | [] => none
 | a :: xs => match latest xs with
   | some v => some v
   | none => if a = 0 then none else some a
def selected (initial : Nat) (xs : List Nat) := (latest xs).getD initial
theorem fold_latest (xs : List Nat) (initial : Nat) :
 xs.foldl step initial = selected initial xs := by
 induction xs generalizing initial with
 | nil => rfl
 | cons a xs ih =>
   simp only [List.foldl_cons, ih, selected, latest]
   cases h : latest xs with
   | some v => rfl
   | none => by_cases ha : a = 0 <;> simp [step, ha]
theorem append_noop (xs : List Nat) (initial : Nat) :
 (xs ++ [0]).foldl step initial = xs.foldl step initial := by simp [List.foldl_append, step]
theorem append_write (xs : List Nat) (initial a : Nat) (ha : a ≠ 0) :
 (xs ++ [a]).foldl step initial = a := by simp [List.foldl_append, step, ha]
#print axioms fold_latest
#print axioms append_noop
#print axioms append_write
end Carry
