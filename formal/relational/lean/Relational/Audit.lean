/-! Concrete positive witnesses complement safety-only statements. -/
import Relational.Paged
import Relational.Programs
import Relational.Mixture

namespace Relational

def auditHistory : List Occurrence :=
  [⟨6, 0, 0⟩, ⟨5, 0, 0⟩, ⟨4, 0, 0⟩, ⟨3, 0, 0⟩, ⟨2, 0, 0⟩, ⟨1, 0, 0⟩]

theorem audit_page_positive :
    (lookupPage 2 2 7 0 0 auditHistory).map (fun hit => hit.record.position) = [2, 1] := by
  decide

theorem audit_current_excluded :
    (lookupPage 2 0 6 0 0 auditHistory).map (fun hit => hit.record.position) = [5, 4] := by
  decide

theorem audit_page_outside :
    (lookupPage 2 3 7 0 0 auditHistory).map (fun hit => hit.record.position) = [] := by
  decide

theorem audit_positive_cap_slot (rank cap : Nat) (positiveCap : 0 < cap) :
    pageSlot rank cap < cap := by
  exact Nat.mod_lt rank positiveCap

theorem audit_mixture_positive : mixedMass 3 1 [(2, 0), (0, 2)] = 8 := by
  decide

theorem audit_zero_memory_weight : mixedMass 1 0 [(2, 0), (0, 2)] = 2 := by
  decide

end Relational
