import Std
namespace TypedMemory
inductive Cell (V : Type) where
 | absent : Cell V
 | value : V → Cell V
 | ref : Nat → Cell V
 deriving DecidableEq, Repr
abbrev Memory (V : Type) := Nat → Cell V

def put (m : Memory V) (i : Nat) (c : Cell V) : Memory V :=
 fun j => if j = i then c else m j

def read : Nat → Memory V → Nat → Option V
 | 0, _, _ => none
 | fuel+1, m, i => match m i with
   | .absent => none
   | .value v => some v
   | .ref j => read fuel m j

def set (m : Memory V) (i : Nat) (v : V) := put m i (.value v)
def delete (m : Memory V) (i : Nat) := put m i .absent
def link (m : Memory V) (i j : Nat) := put m i (.ref j)
def copy (fuel : Nat) (m : Memory V) (i j : Nat) :=
 put m i (match read fuel m j with | none => .absent | some v => .value v)
def query (fuel : Nat) (m : Memory V) (i : Nat) := (read fuel m i, m)

@[simp] theorem put_same (m : Memory V) (i : Nat) (c : Cell V) : put m i c i = c := by simp [put]
@[simp] theorem put_other (m : Memory V) (i j : Nat) (c : Cell V) (h : j ≠ i) : put m i c j = m j := by simp [put, h]
theorem set_read (m : Memory V) (i fuel : Nat) (v : V) : read (fuel+1) (set m i v) i = some v := by simp [read, set]
theorem delete_read (m : Memory V) (i fuel : Nat) : read (fuel+1) (delete m i) i = none := by simp [read, delete]
theorem set_detaches (m : Memory V) (i : Nat) (v : V) : set m i v i = .value v := by simp [set]
theorem link_live (m : Memory V) (i j fuel : Nat) : read (fuel+1) (link m i j) i = read fuel (link m i j) j := by simp [read, link]
theorem copy_snapshot (m : Memory V) (i j fuel budget : Nat) : read (budget+1) (copy fuel m i j) i = read fuel m j := by
 unfold copy
 cases h : read fuel m j <;> simp [read]
theorem snapshot_survives (m : Memory V) (i j k fuel budget : Nat) (c : Cell V) (h : i ≠ k) :
 read (budget+1) (put (copy fuel m i j) k c) i = read fuel m j := by
 unfold copy
 cases hh : read fuel m j <;> simp [read, h]
theorem query_readonly (m : Memory V) (i fuel : Nat) : (query fuel m i).2 = m := rfl
theorem write_noninterference (m : Memory V) (i j : Nat) (c : Cell V) (h : j ≠ i) : put m i c j = m j := put_other m i j c h
theorem overwrite_erases_old (m₁ m₂ : Memory V) (i fuel : Nat) (v : V) :
 read (fuel+1) (set m₁ i v) i = read (fuel+1) (set m₂ i v) i := by simp [set_read]
theorem self_cycle_unknown (m : Memory V) (i fuel : Nat) : read fuel (link m i i) i = none := by
 induction fuel with
 | zero => rfl
 | succ n ih => simpa [read, link] using ih

def empty : Memory Nat := fun _ => .absent
example : read 4 (set (link (copy 4 (set empty 0 7) 1 0) 2 0) 0 9) 1 = some 7 := by decide
example : read 4 (set (link (copy 4 (set empty 0 7) 1 0) 2 0) 0 9) 2 = some 9 := by decide
example : read 4 (delete (link (set empty 0 7) 2 0) 0) 2 = none := by decide
example : read 4 (set (delete (link (set empty 0 7) 2 0) 0) 0 9) 2 = some 9 := by decide
#print axioms set_read
#print axioms delete_read
#print axioms set_detaches
#print axioms link_live
#print axioms copy_snapshot
#print axioms snapshot_survives
#print axioms query_readonly
#print axioms write_noninterference
#print axioms overwrite_erases_old
#print axioms self_cycle_unknown
end TypedMemory
