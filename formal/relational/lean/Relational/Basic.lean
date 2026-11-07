/-!
LRC-003: a dependency-light Lean 4 specification of indexed causal reads
and compositional functional graph programs.

Important: the source has NOT been checked by the Lean kernel in this environment
because Lean/Lake is not installed. It requires `lake env lean ...` in CI.
-/
import Std

namespace Relational

structure Occurrence where
  position : Nat
  relation : Nat
  address : Nat
  deriving Repr, DecidableEq

/-- A read result carries proofs of its causal and address conditions. -/
structure Hit (t r a : Nat) where
  record : Occurrence
  earlier : record.position < t
  sameRelation : record.relation = r
  sameAddress : record.address = a

/-- History is newest-first, matching the Python candidate ordering. -/
def eligible (t r a : Nat) (history : List Occurrence) : List (Hit t r a) :=
  history.filterMap fun w =>
    if hp : w.position < t then
      if hr : w.relation = r then
        if ha : w.address = a then
          some ⟨w, hp, hr, ha⟩
        else none
      else none
    else none

/-- Return at most `budget` eligible records. -/
def lookup (budget t r a : Nat) (history : List Occurrence) : List (Hit t r a) :=
  (eligible t r a history).take budget

/-- By construction every hit is strictly earlier than the query. -/
theorem lookup_causal {budget t r a : Nat} {history : List Occurrence}
    (hit : Hit t r a) (membership : hit ∈ lookup budget t r a history) :
    hit.record.position < t := by
  exact hit.earlier

/-- By construction the read does not cross namespaces. -/
theorem lookup_namespace {budget t r a : Nat} {history : List Occurrence}
    (hit : Hit t r a) (membership : hit ∈ lookup budget t r a history) :
    hit.record.relation = r := by
  exact hit.sameRelation

/-- By construction the read does not cross addresses. -/
theorem lookup_address {budget t r a : Nat} {history : List Occurrence}
    (hit : Hit t r a) (membership : hit ∈ lookup budget t r a history) :
    hit.record.address = a := by
  exact hit.sameAddress

/-- Any future record is ignored by the candidate constructor. -/
theorem future_write_invisible {budget t r a : Nat} (history : List Occurrence)
    (w : Occurrence) (notEarlier : ¬ w.position < t) :
    lookup budget t r a (w :: history) = lookup budget t r a history := by
  simp [lookup, eligible, notEarlier]

/-- A type-independent cap lemma, by induction on the cap. -/
theorem length_take_bounded {α : Type} (n : Nat) (xs : List α) :
    (xs.take n).length ≤ n := by
  induction n generalizing xs with
  | zero => simp
  | succ n ih =>
      cases xs with
      | nil => simp
      | cons x rest =>
          simpa [List.take] using Nat.succ_le_succ (ih rest)

theorem lookup_budget {budget t r a : Nat} (history : List Occurrence) :
    (lookup budget t r a history).length ≤ budget := by
  exact length_take_bounded budget (eligible t r a history)

/-- Typed functional relation semantics for exact multi-hop execution. -/
def follow {Entity Relation : Type} (edge : Relation → Entity → Entity) :
    List Relation → Entity → Entity
  | [], x => x
  | r :: rs, x => follow edge rs (edge r x)

/-- Ordered paths compose; no mistaken reversal of operand order. -/
theorem follow_append {Entity Relation : Type}
    (edge : Relation → Entity → Entity) (p q : List Relation) (x : Entity) :
    follow edge (p ++ q) x = follow edge q (follow edge p x) := by
  induction p generalizing x with
  | nil => rfl
  | cons r rs ih =>
      simp [follow, ih]

/-- Composition is associative, allowing a balanced parallel reduction. -/
def compose {Entity : Type} (f g : Entity → Entity) : Entity → Entity :=
  fun x => g (f x)

theorem compose_assoc {Entity : Type} (f g h : Entity → Entity) :
    compose (compose f g) h = compose f (compose g h) := by
  funext x
  rfl

/-- Compile a sequence of relation functions to a single successor function. -/
def compile {Entity Relation : Type} (edge : Relation → Entity → Entity) :
    List Relation → Entity → Entity
  | [] => fun x => x
  | r :: rs => compose (edge r) (compile edge rs)

/-- Compilation and dependent pointer chasing have identical semantics. -/
theorem compile_correct {Entity Relation : Type}
    (edge : Relation → Entity → Entity) (path : List Relation) (x : Entity) :
    compile edge path x = follow edge path x := by
  induction path generalizing x with
  | nil => rfl
  | cons r rs ih =>
      simp [compile, compose, follow, ih]

/-- Identity is a neutral element for either side of the composition. -/
theorem compose_identity_left {Entity : Type} (f : Entity → Entity) :
    compose (fun x => x) f = f := by
  funext x
  rfl

theorem compose_identity_right {Entity : Type} (f : Entity → Entity) :
    compose f (fun x => x) = f := by
  funext x
  rfl

end Relational
