import Std

/-!
COG-UNIFIED-001: abstract contracts, not a neural implementation proof.

No theorem below asserts learned semantic correctness, optimizer convergence,
universal retrieval, universal no-forgetting, wall-clock speed, or refinement
of any Python/PyTorch program.  The hypotheses are part of each guarantee.
-/

namespace COGUnified

abbrev Store (Key Value : Type) := Key → Value

def write [DecidableEq Key] (s : Store Key Value) (key : Key)
    (value : Value) : Store Key Value :=
  fun k => if k = key then value else s k

theorem write_read_same [DecidableEq Key] (s : Store Key Value)
    (key : Key) (value : Value) : write s key value key = value := by
  simp [write]

theorem write_read_other [DecidableEq Key] (s : Store Key Value)
    (key other : Key) (value : Value) (h : other ≠ key) :
    write s key value other = s other := by
  simp [write, h]

theorem write_overwrite [DecidableEq Key] (s : Store Key Value)
    (key : Key) (old new : Value) :
    write (write s key old) key new = write s key new := by
  funext k
  by_cases h : k = key <;> simp [write, h]

theorem distinct_writes_commute [DecidableEq Key] (s : Store Key Value)
    (a b : Key) (va vb : Value) (hab : a ≠ b) :
    write (write s a va) b vb = write (write s b vb) a va := by
  funext k
  by_cases ha : k = a
  · subst k
    simp [write, hab]
  · by_cases hb : k = b
    · subst k
      simp [write, ha]
    · simp [write, ha, hb]

def gatedCommit (accept : Bool) (before proposed : State) : State :=
  if accept then proposed else before

theorem rejected_commit_preserves (before proposed : State) :
    gatedCommit false before proposed = before := by
  rfl

theorem accepted_commit_uses_proposal (before proposed : State) :
    gatedCommit true before proposed = proposed := by
  rfl

/- A semantic theorem must assume correct grounding; a perfect store alone
   cannot establish that a learned key or learned decoder is right. -/
theorem grounded_read_is_correct (s : Store Key Value)
    (predicted actual : Key) (value : Value)
    (decode : Value → Answer) (answer : Answer)
    (grounding : predicted = actual)
    (stored : s actual = value) (decoding : decode value = answer) :
    decode (s predicted) = answer := by
  rw [grounding, stored, decoding]

/- Exact protection applies to the declared stored cells. -/
theorem protected_cells_preserved [DecidableEq Key]
    (s : Store Key Value) (protectedSet : Key → Prop)
    (key : Key) (value : Value) (notProtected : ¬ protectedSet key) :
    ∀ k, protectedSet k → write s key value k = s k := by
  intro k hk
  apply write_read_other
  intro heq
  subst k
  exact notProtected hk

/- Version counters are mathematical natural numbers: no overflow or reuse. -/
structure Cell (Value : Type) where
  version : Nat
  payload : Value
  deriving Repr

def writeVersioned [DecidableEq Key] (s : Store Key (Cell Value))
    (key : Key) (value : Value) : Store Key (Cell Value) :=
  write s key ⟨(s key).version + 1, value⟩

theorem written_version_strictly_increases [DecidableEq Key]
    (s : Store Key (Cell Value)) (key : Key) (value : Value) :
    (s key).version < (writeVersioned s key value key).version := by
  simp [writeVersioned, write]

theorem version_monotone [DecidableEq Key]
    (s : Store Key (Cell Value)) (key : Key) (value : Value) :
    ∀ k, (s k).version ≤ (writeVersioned s key value k).version := by
  intro k
  by_cases h : k = key
  · subst k
    simp [writeVersioned, write]
  · simp [writeVersioned, write, h]

def SameOn (deps : List Key) (s t : Store Key Value) : Prop :=
  ∀ key, key ∈ deps → s key = t key

def DependsOn (deps : List Key) (reader : Store Key Value → Answer) : Prop :=
  ∀ s t, SameOn deps s t → reader s = reader t

def LocallyDependsOn (deps : List Key) (reader : Store Key Value → Answer)
    (before : Store Key Value) : Prop :=
  ∀ after, SameOn deps before after → reader before = reader after

/- `versionSound` is an explicit history-integrity obligation: equal version
   identifiers must denote equal payloads. A counter/receipt alone is not truth. -/
theorem dependency_cache_sound
    (deps : List Key) (before after : Store Key (Cell Value))
    (reader : Store Key Value → Answer)
    (depends : DependsOn deps reader)
    (versionsEqual : ∀ key, key ∈ deps →
      (before key).version = (after key).version)
    (versionSound : ∀ key, key ∈ deps →
      (before key).version = (after key).version →
      (before key).payload = (after key).payload) :
    reader (fun k => (before k).payload) =
      reader (fun k => (after k).payload) := by
  apply depends
  intro key hkey
  exact versionSound key hkey (versionsEqual key hkey)

theorem cache_requires_same_reader_epoch
    (reader : Nat → Store Key Value → Answer)
    (oldEpoch newEpoch : Nat) (before after : Store Key Value)
    (deps : List Key)
    (epochSame : oldEpoch = newEpoch)
    (depends : DependsOn deps (reader oldEpoch))
    (unchanged : SameOn deps before after) :
    reader oldEpoch before = reader newEpoch after := by
  rw [← epochSame]
  exact depends before after unchanged

/- Local dependence is sufficient for a cache created at one fixed snapshot.
   It does not require every possible query snapshot to use the same keys. -/
theorem local_dependency_cache_sound
    (deps : List Key) (before after : Store Key (Cell Value))
    (reader : Store Key Value → Answer)
    (depends : LocallyDependsOn deps reader (fun k => (before k).payload))
    (versionsEqual : ∀ key, key ∈ deps →
      (before key).version = (after key).version)
    (versionSound : ∀ key, key ∈ deps →
      (before key).version = (after key).version →
      (before key).payload = (after key).payload) :
    reader (fun k => (before k).payload) =
      reader (fun k => (after k).payload) := by
  apply depends
  intro key hkey
  exact versionSound key hkey (versionsEqual key hkey)

theorem local_epoch_cache_sound
    (deps : List Key) (before after : Store Key (Cell Value))
    (reader : Nat → Store Key Value → Answer)
    (oldEpoch newEpoch : Nat) (epochSame : oldEpoch = newEpoch)
    (depends : LocallyDependsOn deps (reader oldEpoch)
      (fun k => (before k).payload))
    (versionsEqual : ∀ key, key ∈ deps →
      (before key).version = (after key).version)
    (versionSound : ∀ key, key ∈ deps →
      (before key).version = (after key).version →
      (before key).payload = (after key).payload) :
    reader oldEpoch (fun k => (before k).payload) =
      reader newEpoch (fun k => (after k).payload) := by
  rw [← epochSame]
  exact local_dependency_cache_sound deps before after (reader oldEpoch)
    depends versionsEqual versionSound

/- Whole-output preservation requires the reader's transitive dependency set,
   rather than merely the output record's own address. -/
theorem dependency_preserving_write [DecidableEq Key]
    (s : Store Key Value) (deps : List Key)
    (reader : Store Key Value → Answer)
    (depends : DependsOn deps reader)
    (key : Key) (value : Value) (outside : key ∉ deps) :
    reader (write s key value) = reader s := by
  apply depends
  intro k hk
  apply write_read_other
  intro heq
  subst k
  exact outside hk

/- Affine scan laws over exact natural-number arithmetic. These are not
   floating-point associativity claims or a claim about state-dependent gates. -/
structure Affine where
  scale : Nat
  shift : Nat
  deriving Repr, DecidableEq

def applyAffine (f : Affine) (s : Nat) : Nat := f.scale * s + f.shift

def compose (f g : Affine) : Affine :=
  ⟨f.scale * g.scale, f.scale * g.shift + f.shift⟩

def identityAffine : Affine := ⟨1, 0⟩

theorem affine_composition_action (f g : Affine) (s : Nat) :
    applyAffine (compose f g) s = applyAffine f (applyAffine g s) := by
  simp [applyAffine, compose, Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

theorem affine_composition_associative (f g h : Affine) :
    compose (compose f g) h = compose f (compose g h) := by
  cases f
  cases g
  cases h
  simp [compose, Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

theorem affine_left_identity (f : Affine) : compose identityAffine f = f := by
  cases f
  simp [compose, identityAffine]

theorem affine_right_identity (f : Affine) : compose f identityAffine = f := by
  cases f
  simp [compose, identityAffine]

/- A pure causal fold has compositional execution and pre causality. These
   properties require an implementation that never reads unseen suffixes. -/
def run (step : State → Token → State) (initial : State) : List Token → State
  | [] => initial
  | x :: xs => run step (step initial x) xs

theorem run_append (step : State → Token → State) (s : State)
    (pre suffix : List Token) :
    run step s (pre ++ suffix) = run step (run step s pre) suffix := by
  induction pre generalizing s with
  | nil => rfl
  | cons x xs ih =>
      simpa [run] using ih (step s x)

/- Finite-run correctness follows from a one-step simulation relation and
   related initial states. No semantic simulation hypothesis is discharged
   here for a trained parser, a neural decoder, or the Python reference. -/
theorem run_preserves_simulation
    (actualStep : Actual → Token → Actual)
    (idealStep : Ideal → Token → Ideal)
    (related : Actual → Ideal → Prop)
    (oneStep : ∀ a i x, related a i →
      related (actualStep a x) (idealStep i x))
    (a : Actual) (i : Ideal) (initial : related a i) (tokens : List Token) :
    related (run actualStep a tokens) (run idealStep i tokens) := by
  induction tokens generalizing a i with
  | nil => exact initial
  | cons x xs ih =>
      exact ih (actualStep a x) (idealStep i x) (oneStep a i x initial)

theorem simulated_run_has_same_readout
    (actualStep : Actual → Token → Actual)
    (idealStep : Ideal → Token → Ideal)
    (actualRead : Actual → Answer) (idealRead : Ideal → Answer)
    (related : Actual → Ideal → Prop)
    (oneStep : ∀ a i x, related a i →
      related (actualStep a x) (idealStep i x))
    (readAgrees : ∀ a i, related a i → actualRead a = idealRead i)
    (a : Actual) (i : Ideal) (initial : related a i) (tokens : List Token) :
    actualRead (run actualStep a tokens) = idealRead (run idealStep i tokens) := by
  exact readAgrees _ _ (run_preserves_simulation actualStep idealStep related
    oneStep a i initial tokens)

def prefixOutput (step : State → Token → State) (read : State → Answer)
    (s : State) (n : Nat) (input : List Token) : Answer :=
  read (run step s (input.take n))

theorem prefix_output_independent_of_suffix
    (step : State → Token → State) (read : State → Answer)
    (s : State) (pre leftSuffix rightSuffix : List Token) :
    prefixOutput step read s pre.length (pre ++ leftSuffix) =
      prefixOutput step read s pre.length (pre ++ rightSuffix) := by
  simp [prefixOutput, List.take_append]

/- Restricted residual adapters preserve all inputs in a protectedSet domain only
   when their residual is zero on that whole domain, not merely sampled tests. -/
def edited (base residual : Input → Nat) (x : Input) : Nat :=
  base x + residual x

theorem edit_preserves_iff_zero (base residual : Input → Nat) (x : Input) :
    edited base residual x = base x ↔ residual x = 0 := by
  simp [edited]

theorem protected_domain_preserved
    (base residual : Input → Nat) (protectedSet : Input → Prop)
    (zeroOnDomain : ∀ x, protectedSet x → residual x = 0) :
    ∀ x, protectedSet x → edited base residual x = base x := by
  intro x hx
  simp [edited, zeroOnDomain x hx]

def routedRepair (route : Input → Bool) (old repaired : Input → Output)
    (x : Input) : Output := if route x then repaired x else old x

theorem isolated_adapter_preserves_outside_route
    (route : Input → Bool) (old repaired : Input → Output)
    (x : Input) (outside : route x = false) :
    routedRepair route old repaired x = old x := by
  simp [routedRepair, outside]

theorem isolated_adapter_preserves_declared_domain
    (route : Input → Bool) (old repaired : Input → Output)
    (domain : Input → Prop)
    (outside : ∀ x, domain x → route x = false) :
    ∀ x, domain x → routedRepair route old repaired x = old x := by
  intro x hx
  exact isolated_adapter_preserves_outside_route route old repaired x (outside x hx)

/- Counterexample 1: finite regression agreement does not prove universal
   preservation, even for a deterministic one-scalar output. -/
def finiteRegressionEdit (limit x : Nat) : Nat :=
  if x = limit + 1 then 1 else 0

theorem finite_regression_passes (limit : Nat) :
    ∀ x, x ≤ limit → finiteRegressionEdit limit x = 0 := by
  intro x hx
  have hne : x ≠ limit + 1 := by omega
  simp [finiteRegressionEdit, hne]

theorem finite_regression_still_changes_output (limit : Nat) :
    finiteRegressionEdit limit (limit + 1) ≠ 0 := by
  simp [finiteRegressionEdit]

/- Counterexample 2: preserving a LINK cell is not preserving its live result. -/
inductive Record where
  | value : Nat → Record
  | link : Nat → Record
  | deleted : Record
  deriving Repr, DecidableEq

def resolve : Nat → Store Nat Record → Nat → Option Nat
  | 0, _, _ => none
  | fuel + 1, s, key =>
      match s key with
      | .value v => some v
      | .link target => resolve fuel s target
      | .deleted => none

def readDependencies : Nat → Store Nat Record → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, s, key => key ::
      match s key with
      | .link target => readDependencies fuel s target
      | _ => []

/- For this exact deterministic bounded LINK interpreter, all cells on the
   actual old path suffice. This theorem does not cover a neural selector,
   hidden recurrent state, candidate-index changes, or a general program. -/
theorem unchanged_read_path_preserves_resolution
    (fuel : Nat) (before after : Store Nat Record) (key : Nat)
    (unchanged : SameOn (readDependencies fuel before key) before after) :
    resolve fuel before key = resolve fuel after key := by
  induction fuel generalizing key with
  | zero => rfl
  | succ n ih =>
      have headSame : before key = after key :=
        unchanged key (by simp [readDependencies])
      cases h : before key with
      | value value => simp [resolve, ← headSame, h]
      | deleted => simp [resolve, ← headSame, h]
      | link target =>
          have tailSame : SameOn (readDependencies n before target) before after := by
            intro k hk
            exact unchanged k (by simp [readDependencies, h, hk])
          simpa only [resolve, ← headSame, h] using ih target tailSame

theorem bounded_link_reader_locally_depends_on_path
    (fuel : Nat) (before : Store Nat Record) (key : Nat) :
    LocallyDependsOn (readDependencies fuel before key)
      (fun s => resolve fuel s key) before := by
  intro after unchanged
  exact unchanged_read_path_preserves_resolution fuel before after key unchanged

def setRecord (s : Store Nat Record) (key value : Nat) : Store Nat Record :=
  write s key (.value value)

def deleteRecord (s : Store Nat Record) (key : Nat) : Store Nat Record :=
  write s key .deleted

def linkRecord (s : Store Nat Record) (destination source : Nat) : Store Nat Record :=
  write s destination (.link source)

def snapshotCopy (s : Store Nat Record) (fuel source destination : Nat) :
    Option (Store Nat Record) :=
  match resolve fuel s source with
  | none => none
  | some value => some (write s destination (.value value))

theorem set_record_reads_value (s : Store Nat Record) (fuel key value : Nat) :
    resolve (fuel + 1) (setRecord s key value) key = some value := by
  simp [resolve, setRecord, write]

theorem deleted_record_reads_none (s : Store Nat Record) (fuel key : Nat) :
    resolve (fuel + 1) (deleteRecord s key) key = none := by
  simp [resolve, deleteRecord, write]

theorem delete_preserves_other_cell (s : Store Nat Record) (key other : Nat)
    (different : other ≠ key) : deleteRecord s key other = s other := by
  simp [deleteRecord, write, different]

theorem live_link_dereferences_current_state (s : Store Nat Record)
    (fuel destination source : Nat) :
    resolve (fuel + 1) (linkRecord s destination source) destination =
      resolve fuel (linkRecord s destination source) source := by
  simp [resolve, linkRecord, write]

theorem snapshot_copy_materializes_value (s : Store Nat Record)
    (fuel source destination value : Nat)
    (sourceResolved : resolve fuel s source = some value) :
    snapshotCopy s fuel source destination =
      some (write s destination (.value value)) := by
  simp [snapshotCopy, sourceResolved]

theorem snapshot_copy_refuses_unresolved (s : Store Nat Record)
    (fuel source destination : Nat)
    (sourceUnresolved : resolve fuel s source = none) :
    snapshotCopy s fuel source destination = none := by
  simp [snapshotCopy, sourceUnresolved]

theorem snapshot_value_survives_other_write (s : Store Nat Record)
    (fuel destination other value : Nat) (replacement : Record)
    (different : destination ≠ other) :
    resolve (fuel + 1)
      (write (write s destination (.value value)) other replacement)
      destination = some value := by
  simp [resolve, write, different]

def linkedBefore : Store Nat Record :=
  fun key => if key = 0 then .link 1 else .value 5

def linkedAfter : Store Nat Record := write linkedBefore 1 (.value 7)

theorem link_cell_preserved : linkedBefore 0 = linkedAfter 0 := by
  rfl

theorem live_link_answer_changed :
    resolve 2 linkedBefore 0 = some 5 ∧ resolve 2 linkedAfter 0 = some 7 := by
  constructor <;> rfl

def readCount : Nat → Store Nat Record → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, s, key =>
      match s key with
      | .value _ => 1
      | .link target => 1 + readCount fuel s target
      | .deleted => 1

theorem link_reads_bounded_by_fuel (fuel : Nat) (s : Store Nat Record)
    (key : Nat) : readCount fuel s key ≤ fuel := by
  induction fuel generalizing key with
  | zero => simp [readCount]
  | succ n ih =>
      cases h : s key with
      | value v => simp [readCount, h]
      | deleted => simp [readCount, h]
      | link target =>
          simp only [readCount, h]
          have bound := ih target
          omega

/- Counterexample 3: distinct write destinations do not suffice for parallel
   equivalence when one transaction reads a cell the other writes. -/
def readDependentA (s : Store Nat Nat) : Store Nat Nat :=
  write s 0 (s 1 + 1)

def readDependentB (s : Store Nat Nat) : Store Nat Nat :=
  write s 1 (s 0 + 1)

theorem disjoint_targets_still_interfere :
    readDependentA (readDependentB (fun _ => 0)) 0 = 2 ∧
    readDependentB (readDependentA (fun _ => 0)) 0 = 1 := by
  constructor <;> rfl

/- Counterexample 4: an exact reader cannot retrieve an omitted candidate. -/
def candidateRead [DecidableEq Key] (s : Store Key Value)
    (candidates : List Key) (key : Key) : Option Value :=
  if key ∈ candidates then some (s key) else none

theorem omitted_candidate_is_unread [DecidableEq Key]
    (s : Store Key Value) (candidates : List Key) (key : Key)
    (omitted : key ∉ candidates) : candidateRead s candidates key = none := by
  simp [candidateRead, omitted]

theorem concrete_sparse_miss :
    candidateRead (fun _ : Nat => 42) [0, 1] 2 = none := by
  decide

theorem three_histories_two_states_collide (encode : Fin 3 → Bool) :
    encode 0 = encode 1 ∨ encode 0 = encode 2 ∨ encode 1 = encode 2 := by
  cases h0 : encode 0 <;> cases h1 : encode 1 <;> cases h2 : encode 2 <;>
    simp_all

/- The core obstruction behind finite-memory exact recall: indistinguishable
   encoded histories with different answers cannot both decode correctly.
   A separate counting argument is needed to force such a collision. -/
theorem encoding_collision_blocks_exact_recall
    (encode : History → State) (decode : State → Query → Answer)
    (truth : History → Query → Answer) (h₁ h₂ : History) (q : Query)
    (collision : encode h₁ = encode h₂)
    (different : truth h₁ q ≠ truth h₂ q) :
    ¬ (decode (encode h₁) q = truth h₁ q ∧
       decode (encode h₂) q = truth h₂ q) := by
  intro both
  apply different
  calc
    truth h₁ q = decode (encode h₁) q := both.1.symm
    _ = decode (encode h₂) q := by rw [collision]
    _ = truth h₂ q := both.2

end COGUnified
