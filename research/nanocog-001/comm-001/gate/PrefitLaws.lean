import Lean

/-! Structural properties only. No assertion that a learned decoder satisfies
the local-input premises, or that its initialized scratch discovers evidence. -/

theorem identical_view_cannot_answer_both
    {View Answer : Type}
    (predict : View → Answer) (x y : View) (a b : Answer)
    (same : x = y) (opposed : a ≠ b) :
    ¬ (predict x = a ∧ predict y = b) := by
  rintro ⟨ha, hb⟩
  exact opposed (ha.symm.trans ((congrArg predict same).trans hb))

def ntrue (a b : Bool) : Nat :=
  (if a then 1 else 0) + (if b then 1 else 0)

theorem identical_view_at_most_one_correct
    {View Answer : Type} [DecidableEq Answer]
    (predict : View → Answer) (x y : View) (a b : Answer)
    (same : x = y) (opposed : a ≠ b) :
    ntrue (decide (predict x = a)) (decide (predict y = b)) ≤ 1 := by
  by_cases h : predict x = a
  · have k : predict y ≠ b := by
      intro hb
      exact identical_view_cannot_answer_both predict x y a b same opposed ⟨h, hb⟩
    simp [ntrue, h, k]
  · simp [ntrue, h]
    split <;> omega

theorem route_attrition_counts_are_nested
    (worlds published resident selected : Nat)
    (h₁ : published ≤ worlds) (h₂ : resident ≤ published)
    (h₃ : selected ≤ resident) :
    (worlds - published) + (published - resident) +
      (resident - selected) + selected = worlds := by
  omega
