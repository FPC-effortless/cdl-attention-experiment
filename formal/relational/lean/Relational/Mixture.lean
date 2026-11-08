/-!
Exact finite-mass mixture contract, added for the mathematical review.
This is not a verification of an unavailable floating-point byte-cache model.
Natural masses and nonnegative integer weights represent rational probabilities
after division by the positive common denominator (a + b) * mass.
-/
import Std

namespace Relational

def mixedMass (a b : Nat) : List (Nat × Nat) → Nat
  | [] => 0
  | cell :: rest => a * cell.1 + b * cell.2 + mixedMass a b rest

theorem mixedMass_total (a b : Nat) (cells : List (Nat × Nat)) :
    mixedMass a b cells =
      a * (cells.map Prod.fst).sum + b * (cells.map Prod.snd).sum := by
  induction cells with
  | nil => simp [mixedMass]
  | cons cell rest ih =>
      simp only [mixedMass, List.map_cons, List.sum_cons, Nat.mul_add]
      rw [ih]
      omega

theorem mixture_normalized_mass (a b mass : Nat) (cells : List (Nat × Nat))
    (baseNormalized : (cells.map Prod.fst).sum = mass)
    (memoryNormalized : (cells.map Prod.snd).sum = mass) :
    mixedMass a b cells = (a + b) * mass := by
  rw [mixedMass_total, baseNormalized, memoryNormalized, Nat.add_mul]

theorem mixture_denominator_positive (a b mass : Nat)
    (weightPositive : 0 < a + b) (massPositive : 0 < mass) :
    0 < (a + b) * mass := Nat.mul_pos weightPositive massPositive

end Relational
