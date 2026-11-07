/-!
LRC-V4: a total denotational specification for paged causal history.

This file is intentionally independent of neural training. In particular,
reachability of a prior record by an ordinal page address DOES NOT imply that
an inference policy can learn its correct address from language.

NOTE: Lean toolchain is not installed in the present environment. Statements
are formalization candidates; they have not been kernel-checked here.
-/
import Relational.Basic

namespace Relational

/-- The C-sized page of eligible historical occurrences at an arbitrary depth. -/
def lookupPage (budget page t relation address : Nat)
    (history : List Occurrence) : List (Hit t relation address) :=
  ((eligible t relation address history).drop (page * budget)).take budget

/-- The newest page exactly matches the original capped lookup. -/
theorem lookupPage_zero (budget t relation address : Nat)
    (history : List Occurrence) :
    lookupPage budget 0 t relation address history =
      lookup budget t relation address history := by
  simp [lookupPage, lookup]

/-- Every page has a uniform maximum number of inspected candidates. -/
theorem lookupPage_budget (budget page t relation address : Nat)
    (history : List Occurrence) :
    (lookupPage budget page t relation address history).length ≤ budget := by
  unfold lookupPage
  exact length_take_bounded budget
    ((eligible t relation address history).drop (page * budget))

/-- A page never returns a current or future occurrence. -/
theorem lookupPage_causal {budget page t relation address : Nat}
    {history : List Occurrence} (hit : Hit t relation address)
    (_ : hit ∈ lookupPage budget page t relation address history) :
    hit.record.position < t := hit.earlier

/-- Page results cannot cross relation namespaces. -/
theorem lookupPage_relation {budget page t relation address : Nat}
    {history : List Occurrence} (hit : Hit t relation address)
    (_ : hit ∈ lookupPage budget page t relation address history) :
    hit.record.relation = relation := hit.sameRelation

/-- Page results cannot cross address namespaces. -/
theorem lookupPage_address {budget page t relation address : Nat}
    {history : List Occurrence} (hit : Hit t relation address)
    (_ : hit ∈ lookupPage budget page t relation address history) :
    hit.record.address = address := hit.sameAddress

/-- A record's reverse-chronological rank uniquely splits into page + slot. -/
def pageNumber (rank cap : Nat) : Nat := rank / cap

def pageSlot (rank cap : Nat) : Nat := rank % cap

theorem rank_reconstructed (rank cap : Nat) :
    pageNumber rank cap * cap + pageSlot rank cap = rank := by
  simpa [pageNumber, pageSlot, Nat.mul_comm, Nat.add_comm]
    using Nat.mod_add_div rank cap

/-- No new future occurrence can affect any fixed page of a causal prefix. -/
theorem lookupPage_future_invisible
    (budget page t relation address : Nat) (history : List Occurrence)
    (w : Occurrence) (notEarlier : ¬ w.position < t) :
    lookupPage budget page t relation address (w :: history) =
    lookupPage budget page t relation address history := by
  simp [lookupPage, eligible, notEarlier]

end Relational
