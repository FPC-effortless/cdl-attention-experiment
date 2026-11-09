import Std

/-!
# Structural obligations for the initial regulated CSM ablation

This file is a dependency-light specification, not a complete formalization of
the numerical CSM implementation. The concrete candidate retains the original
CSM transition, readout, cursor, and link semantics and adds an exogenous trace
to its update proposal. Its real-valued equations are analyzed separately.

The generic `Core` below contains the ORIGINAL CSM state. No new action schema,
slot-reuse policy, generation tags, or microstep mechanism is introduced here.

The theorems concern pure transitions, an explicit neutral regulatory branch,
selected payload writes, causal list rollouts, independent batch lanes, and
commutation of writes whose payloads were frozen from one common snapshot.
They do not establish stability of tanh, gradients, learned routes, semantic
correctness, or agreement with a particular floating-point implementation.

STATUS: compiler-checked with Lean 4.19.0 and --trust=0. See FORMAL_SCOPE.txt
for the command, scope, and distinction from concrete numerical verification.
-/

set_option autoImplicit false

universe u v w x y z

namespace CSMRegulation

abbrev Store (Index : Type u) (Value : Type v) := Index -> Value

section SelectedWrites

variable {Index : Type u} {Value : Type v} [DecidableEq Index]

/-- A single payload update. The supplied value has already been computed. -/
def overwrite (memory : Store Index Value) (slot : Index) (value : Value) :
    Store Index Value :=
  fun probe => if probe = slot then value else memory probe

@[simp] theorem overwrite_selected
    (memory : Store Index Value) (slot : Index) (value : Value) :
    overwrite memory slot value slot = value := by
  simp [overwrite]

/-- Every payload outside the selected slot is unchanged. -/
theorem overwrite_frame
    (memory : Store Index Value) (slot probe : Index) (value : Value)
    (different : probe ≠ slot) :
    overwrite memory slot value probe = memory probe := by
  simp [overwrite, different]

theorem overwrite_same
    (memory : Store Index Value) (slot : Index) :
    overwrite memory slot (memory slot) = memory := by
  funext probe
  by_cases selected : probe = slot
  · subst probe
    simp [overwrite]
  · simp [overwrite, selected]

/-- Commit a proposal evaluated on `snapshot`, rather than on `current`. -/
def snapshotCommit
    (snapshot current : Store Index Value)
    (slot : Index) (proposal : Store Index Value -> Value) :
    Store Index Value :=
  overwrite current slot (proposal snapshot)

@[simp] theorem snapshotCommit_selected
    (snapshot current : Store Index Value)
    (slot : Index) (proposal : Store Index Value -> Value) :
    snapshotCommit snapshot current slot proposal slot = proposal snapshot := by
  simp [snapshotCommit]

theorem snapshotCommit_frame
    (snapshot current : Store Index Value)
    (slot probe : Index) (proposal : Store Index Value -> Value)
    (different : probe ≠ slot) :
    snapshotCommit snapshot current slot proposal probe = current probe := by
  exact overwrite_frame current slot probe (proposal snapshot) different

/-- Frozen writes to distinct slots commute. -/
theorem overwrite_commute
    (memory : Store Index Value) (left right : Index)
    (leftValue rightValue : Value) (different : left ≠ right) :
    overwrite (overwrite memory left leftValue) right rightValue =
      overwrite (overwrite memory right rightValue) left leftValue := by
  funext probe
  by_cases atLeft : probe = left
  · subst probe
    simp [overwrite, different]
  · by_cases atRight : probe = right
    · subst probe
      simp [overwrite, different, Ne.symm different]
    · simp [overwrite, atLeft, atRight]

/-
Both proposal values below are evaluated against the SAME immutable snapshot.
This is not a theorem about recomputing a state-dependent route or proposal
after the first commit, nor about simultaneously updating shared cursors or
regulators. Those would require additional dependency conditions.
-/
theorem fixed_snapshot_disjoint_commits
    (snapshot current : Store Index Value) (left right : Index)
    (leftProposal rightProposal : Store Index Value -> Value)
    (different : left ≠ right) :
    snapshotCommit snapshot
        (snapshotCommit snapshot current left leftProposal) right rightProposal =
      snapshotCommit snapshot
        (snapshotCommit snapshot current right rightProposal) left leftProposal := by
  exact overwrite_commute current left right
    (leftProposal snapshot) (rightProposal snapshot) different

end SelectedWrites

/--
A frame for one isolated payload operation. `rest` can contain the existing
cursor, links, and other state. This wrapper does not specify their dynamics.
-/
structure FramedStore (Index : Type u) (Value : Type v) (Rest : Type w) where
  payload : Store Index Value
  rest : Rest

section FramedWrites

variable {Index : Type u} {Value : Type v} {Rest : Type w}
variable [DecidableEq Index]

def framedOverwrite
    (state : FramedStore Index Value Rest) (slot : Index) (value : Value) :
    FramedStore Index Value Rest :=
  { state with payload := overwrite state.payload slot value }

@[simp] theorem framedOverwrite_preserves_rest
    (state : FramedStore Index Value Rest) (slot : Index) (value : Value) :
    (framedOverwrite state slot value).rest = state.rest := by
  rfl

theorem framedOverwrite_frame
    (state : FramedStore Index Value Rest) (slot probe : Index) (value : Value)
    (different : probe ≠ slot) :
    (framedOverwrite state slot value).payload probe = state.payload probe := by
  exact overwrite_frame state.payload slot probe value different

end FramedWrites

abbrev Step (State : Type u) (Input : Type v) := State -> Input -> State

section PureTransitions

variable {State : Type u} {Input : Type v}

/-- A disabled optional operation is exactly the identity. -/
def optionalUpdate (enabled : Bool) (operation : State -> State) (state : State) :
    State :=
  match enabled with
  | false => state
  | true => operation state

@[simp] theorem optionalUpdate_disabled
    (operation : State -> State) (state : State) :
    optionalUpdate false operation state = state := by
  rfl

@[simp] theorem optionalUpdate_enabled
    (operation : State -> State) (state : State) :
    optionalUpdate true operation state = operation state := by
  rfl

/-- Standard left-to-right execution of a pure, single-input transition. -/
def rollout (step : Step State Input) (initial : State) (inputs : List Input) :
    State :=
  match inputs with
  | [] => initial
  | input :: tail => rollout step (step initial input) tail

/-- Execute at most `count` inputs, without inspecting the remaining suffix. -/
def rolloutPrefix
    (step : Step State Input) (count : Nat) (initial : State)
    (inputs : List Input) : State :=
  match count, inputs with
  | 0, _ => initial
  | Nat.succ _, [] => initial
  | Nat.succ remaining, input :: tail =>
      rolloutPrefix step remaining (step initial input) tail

theorem rollout_append
    (step : Step State Input) (initial : State)
    (past suffix : List Input) :
    rollout step initial (past ++ suffix) =
      rollout step (rollout step initial past) suffix := by
  induction past generalizing initial with
  | nil => rfl
  | cons input tail inductionHypothesis =>
      change rollout step (step initial input) (tail ++ suffix) =
        rollout step (rollout step (step initial input) tail) suffix
      exact inductionHypothesis (initial := step initial input)

theorem rolloutPrefix_append_length
    (step : Step State Input) (initial : State)
    (past suffix : List Input) :
    rolloutPrefix step past.length initial (past ++ suffix) =
      rollout step initial past := by
  induction past generalizing initial with
  | nil => rfl
  | cons input tail inductionHypothesis =>
      change rolloutPrefix step tail.length (step initial input) (tail ++ suffix) =
        rollout step (step initial input) tail
      exact inductionHypothesis (initial := step initial input)

/-- Equal prefixes produce equal prefix states, regardless of future suffix. -/
theorem causal_prefix_noninterference
    (step : Step State Input) (initial : State)
    (past futureLeft futureRight : List Input) :
    rolloutPrefix step past.length initial (past ++ futureLeft) =
      rolloutPrefix step past.length initial (past ++ futureRight) := by
  simp only [rolloutPrefix_append_length]

end PureTransitions

section ExogenousRegulation

variable {Core : Type u} {Reg : Type v} {Input : Type w}

/-- An explicit neutral trace represents the disabled regulatory branch. -/
def suppliedTrace (enabled : Bool) (neutral actual : Reg) : Reg :=
  match enabled with
  | false => neutral
  | true => actual

/--
Structural ordering of the candidate:
  1. compute the exogenous trace from the old trace and current input;
  2. provide that trace to the original CSM core's proposal pathway;
  3. retain the new trace.

Here `Input` denotes prepared causal token features, including the information
needed from preceding input embeddings. The original frontend may retain its
previous two embeddings in a history field of the implementation's core state.
The abstract independence claim is therefore conditional on the SAME prepared
features/token history, not on arbitrary changes to those history fields.
The feature frontend must use the same initialized boundary in both runs
(zero history for the candidate's reset boundary).

`coreStep` must encode the original cursor/link/readout semantics unchanged.
The disabled branch may continue updating its unused trace; the theorem below
therefore compares CORE states, not equality of the full augmented state.
-/
def regulatedStep
    (enabled : Bool) (neutral : Reg)
    (traceUpdate : Reg -> Input -> Reg)
    (coreStep : Core -> Input -> Reg -> Core) :
    Step (Core × Reg) Input :=
  fun state input =>
    let nextTrace := traceUpdate state.2 input
    (coreStep state.1 input (suppliedTrace enabled neutral nextTrace), nextTrace)

@[simp] theorem regulatedStep_trace_projection
    (enabled : Bool) (neutral : Reg)
    (traceUpdate : Reg -> Input -> Reg)
    (coreStep : Core -> Input -> Reg -> Core)
    (state : Core × Reg) (input : Input) :
    (regulatedStep enabled neutral traceUpdate coreStep state input).2 =
      traceUpdate state.2 input := by
  rfl

/--
For identical prepared causal features and old trace, the next trace is
independent of other core fields. This does not permit changing frontend token
history while pretending that its derived `input` remains unchanged.
-/
theorem exogenous_trace_core_independence
    (enabled : Bool) (neutral : Reg)
    (traceUpdate : Reg -> Input -> Reg)
    (coreStep : Core -> Input -> Reg -> Core)
    (leftCore rightCore : Core) (trace : Reg) (input : Input) :
    (regulatedStep enabled neutral traceUpdate coreStep (leftCore, trace) input).2 =
      (regulatedStep enabled neutral traceUpdate coreStep (rightCore, trace) input).2 := by
  rfl

theorem disabled_regulation_core_step
    (neutral : Reg) (traceUpdate : Reg -> Input -> Reg)
    (coreStep : Core -> Input -> Reg -> Core) (baseline : Step Core Input)
    (neutralIsBaseline : ∀ core input, coreStep core input neutral = baseline core input)
    (state : Core × Reg) (input : Input) :
    (regulatedStep false neutral traceUpdate coreStep state input).1 =
      baseline state.1 input := by
  exact neutralIsBaseline state.1 input

/--
If neutral injection is the baseline transition, disabling regulation recovers
the baseline CORE trajectory for every input list, even if unused traces evolve.
Equality of outputs additionally requires an identical readout depending only
on the identical core and prepared features; it is not asserted separately here.
-/
theorem disabled_regulation_core_rollout
    (neutral : Reg) (traceUpdate : Reg -> Input -> Reg)
    (coreStep : Core -> Input -> Reg -> Core) (baseline : Step Core Input)
    (neutralIsBaseline : ∀ core input, coreStep core input neutral = baseline core input)
    (initial : Core × Reg) (inputs : List Input) :
    (rollout (regulatedStep false neutral traceUpdate coreStep) initial inputs).1 =
      rollout baseline initial.1 inputs := by
  induction inputs generalizing initial with
  | nil => rfl
  | cons input tail inductionHypothesis =>
      change
        (rollout (regulatedStep false neutral traceUpdate coreStep)
          (regulatedStep false neutral traceUpdate coreStep initial input) tail).1 =
        rollout baseline (baseline initial.1 input) tail
      rw [inductionHypothesis]
      change rollout baseline (coreStep initial.1 input neutral) tail =
        rollout baseline (baseline initial.1 input) tail
      rw [neutralIsBaseline]

end ExogenousRegulation

section IndependentBatching

variable {State : Type u} {Input : Type v} {Lane : Type w}

/-- Each lane has its own complete state; there are no shared mutable fields. -/
def batchStep (step : Step State Input) : Step (Lane -> State) (Lane -> Input) :=
  fun states inputs lane => step (states lane) (inputs lane)

@[simp] theorem batchStep_projection
    (step : Step State Input) (states : Lane -> State)
    (inputs : Lane -> Input) (lane : Lane) :
    batchStep step states inputs lane = step (states lane) (inputs lane) := by
  rfl

/-- Changes in other lanes cannot affect the selected lane's next state. -/
theorem batch_lane_independence
    (step : Step State Input)
    (leftStates rightStates : Lane -> State)
    (leftInputs rightInputs : Lane -> Input) (lane : Lane)
    (sameState : leftStates lane = rightStates lane)
    (sameInput : leftInputs lane = rightInputs lane) :
    batchStep step leftStates leftInputs lane =
      batchStep step rightStates rightInputs lane := by
  change step (leftStates lane) (leftInputs lane) =
    step (rightStates lane) (rightInputs lane)
  rw [sameState, sameInput]

/-- Batched execution projects to exactly the independent per-lane rollout. -/
theorem batch_rollout_projection
    (step : Step State Input) (initial : Lane -> State)
    (inputs : List (Lane -> Input)) (lane : Lane) :
    (rollout (batchStep step) initial inputs) lane =
      rollout step (initial lane) (inputs.map (fun input => input lane)) := by
  induction inputs generalizing initial with
  | nil => rfl
  | cons input tail inductionHypothesis =>
      change
        (rollout (batchStep step) (batchStep step initial input) tail) lane =
        rollout step (step (initial lane) (input lane))
          (tail.map (fun nextInput => nextInput lane))
      exact inductionHypothesis (initial := batchStep step initial input)

end IndependentBatching

/-!
## Compact affine-pair algebra

The scalar operations and laws are explicit theorem parameters, not added
axioms. Real-number instances and floating-point error bounds are outside this
Std-only file. The chronological convention is: compose earlier, then later.
-/

structure AffinePair (Scalar : Type u) where
  a : Scalar
  b : Scalar

section AffineComposition

variable {Scalar : Type u}

def affineEval
    (mul add : Scalar -> Scalar -> Scalar)
    (pair : AffinePair Scalar) (state : Scalar) : Scalar :=
  add (mul pair.a state) pair.b

/-- `chronologicalCompose p q` means first apply p, then apply q. -/
def chronologicalCompose
    (mul add : Scalar -> Scalar -> Scalar)
    (earlier later : AffinePair Scalar) : AffinePair Scalar :=
  { a := mul later.a earlier.a
    b := add (mul later.a earlier.b) later.b }

/-- A compact affine pair represents chronological functional composition. -/
theorem chronologicalCompose_eval
    (mul add : Scalar -> Scalar -> Scalar)
    (mulAssoc : ∀ a b c, mul (mul a b) c = mul a (mul b c))
    (leftDistrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c))
    (addAssoc : ∀ a b c, add (add a b) c = add a (add b c))
    (earlier later : AffinePair Scalar) (state : Scalar) :
    affineEval mul add (chronologicalCompose mul add earlier later) state =
      affineEval mul add later (affineEval mul add earlier state) := by
  simp only [affineEval, chronologicalCompose, mulAssoc, leftDistrib, addAssoc]

/--
Chronological affine-pair composition is associative under these three laws.
This permits regrouping a scan of precomputed affine trace transitions; it does
not assert that state-dependent CSM routing itself has this representation.
-/
theorem chronologicalCompose_assoc
    (mul add : Scalar -> Scalar -> Scalar)
    (mulAssoc : ∀ a b c, mul (mul a b) c = mul a (mul b c))
    (leftDistrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c))
    (addAssoc : ∀ a b c, add (add a b) c = add a (add b c))
    (first second third : AffinePair Scalar) :
    chronologicalCompose mul add
        (chronologicalCompose mul add first second) third =
      chronologicalCompose mul add first
        (chronologicalCompose mul add second third) := by
  cases first
  cases second
  cases third
  simp only [chronologicalCompose, mulAssoc, leftDistrib, addAssoc]

end AffineComposition

end CSMRegulation
