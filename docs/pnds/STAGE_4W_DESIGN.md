# PNDS-GATE-001 Stage 4W — Pre-Registration: Write → Boundary → Retrieve

> Registered **before** any implementation, following the ordering that exposed
> both prior defects (`528ad89`, `f989430`). No Stage 4W code is written until
> this document is committed.
>
> Identifier **4W** is deliberate: `PNDS_GATE_001.md` already defines "Stage 4"
> as the verifier. This is not the verifier. Reusing the number would silently
> overwrite a registered experiment ID, which protocol §5 forbids.

## 0. Position in the program

Stage 3b (`STAGE_3B_RESULT_CORRECTED.md`, E4/SUPPORTED) established that a
learned router can **read** key-indexed persistent state and route by it, with
the read causally load-bearing:

\[
S_t \;\rightarrow\; D_\theta \;\rightarrow\; Y_t
\]

The state in Stage 3b is *supplied by the stream generator* and is **present
throughout** the episode. Nothing is ever written. There is no `S_{t+1}` and no
temporal separation. Stage 3b is therefore a substrate result, not a memory
result.

Stage 4W is the smallest experiment that changes that. It asks whether persistent
state can act as a **carrier of information across time**:

\[
X_t \;\xrightarrow{\;\text{Write}_\theta\;}\; S_{t+1}
\;\xrightarrow{\;\boxed{\text{temporal boundary}}\;}\;
\text{Read}_\theta(S_{t+1}, Q_{t+1}) \;\rightarrow\; Y_{t+1}
\]

Promoted, this would support the stronger claim: *an episode-specific piece of
information can be written into persistent state, survive an information
boundary, and causally control a later decision when the original information is
no longer available.*

This is the first experiment in the program with a genuine `S_{t+1}`. It is also
the first where **the information needed after the boundary does not exist in
the post-boundary observation**.

## 1. The central falsification

Two channels can carry the needed information across the boundary:

| channel | path | status |
|---|---|---|
| **intended** | $X_t \rightarrow S_{t+1} \rightarrow Y_{t+1}$ | what the experiment claims |
| **shortcut** | $X_t \rightarrow \theta \;/$ hidden computation $\rightarrow Y_{t+1}$ | invalidates the result |

The shortcut channel is the whole risk. A model that memorises the mapping in
its parameters, or retains `X_t` through any route other than `S_{t+1}`, would
look successful while proving nothing about state.

So the central Stage 4W question is:

> **When information needed after the temporal boundary is removed from the
> current observation, does successful behavior depend specifically on the newly
> written state?**

That is the causal contract, and every gate below is a way of breaking it.

### The parameter-memorisation loophole, closed by construction

If the training distribution contains a stable mapping `X_t → Y_{t+1}`, the
network can learn that mapping directly and needs no state at all. Stage 4W
therefore uses **episode-specific, freshly randomised content** that cannot be
predicted from the parameters:

- Episode A: write "target = 17", boundary, query, answer must be 17
- Episode B: write "target = 4", boundary, query, answer must be 4

With the written content resampled every episode, the only place the answer can
live between the two halves is the state. This is a design constraint, not a
control that is checked afterwards — the environment must be built so that the
weights provably cannot hold the answer.

## 2. The four layers

### Layer 1 — Write representability gate

Per URP v0.4 §34, extended. Do not ask whether training works; ask whether the
mechanism can express the solution at all.

There are now **three** mechanisms rather than one, and the gate must cover all
of them:

1. an ideal **writer** `Write_ideal(X_t) = S*_{t+1} = f(X_t)`;
2. the **persistent state representation** itself;
3. an ideal **reader** `Read_ideal(S_{t+1}, Q_{t+1})`.

Construct `Write_ideal` and `Read_ideal` analytically, wire them through the
**actual** state representation and storage mechanism, and assert that the
correct answer beats the strongest distractor:

\[
\text{Write}_{ideal}(X_t) \rightarrow S_{t+1} \rightarrow
\text{Read}_{ideal}(S_{t+1}, Q_{t+1}) \;\;\text{separates gold.}
\]

This is not the same as Stage 3b's gate. There the gate had one mechanism and
one direction of information flow. Here the state must be *produced* as well as
*consumed*, and a defect in the write path is invisible to a read-only gate —
the reader can score perfectly while nothing was ever written.

The gate must be shown to **fail** on a defective writer (e.g. a writer that
ignores `X_t`), or it guards nothing.

### Layer 2 — Boundary integrity gate

Probably the most important new control. After the boundary, `X_t` must be
genuinely unavailable. The post-boundary observation contains only what the
protocol explicitly permits: `Q_{t+1}` and `S_{t+1}`.

**State ablation — the strong version:**

\[
S_{t+1} \rightarrow \varnothing
\]

If the model still succeeds with the state removed, a leakage channel exists and
the result is **inadmissible**. This is a hard stop, not a degradation.

The ablation must be applied at the level of the *state object*, not at the level
of the features — otherwise a feature-level ablation can leave the information
intact through a path the experimenter did not think to cut.

### Layer 3 — Counterfactual state replacement

Stage 3b's intervention logic carried forward and made sharper.

Let the writer produce `S_{t+1} = W(X_t)`. After the boundary, replace it with
`S'_{t+1} = W(X'_t)` while holding the post-boundary observation **identical**.
The downstream decision must follow `S'`, not the original `X_t`:

\[
do(S_{t+1} \leftarrow S'_{t+1}) \;\;\text{moves the decision as predicted.}
\]

This is stronger than showing that state correlates with success, and it is the
direct analogue of the `do(K \leftarrow K')` intervention that carried Stage 3b's
F2. A correlation that survives replacement is not persistence; it is a
nuisance variable.

### Layer 4 — Writer/read separation

Four arms that decompose the causal chain:

| arm | writer sees `X_t` | state supplied at read | expected accuracy | what it tests |
|---|---|---|---|---|
| **write-only** | yes | destroyed | ≈ chance | nothing is retained without the state |
| **read-only** | no | correct state | ≈ Stage 3b level | the retrieval half alone |
| **write → boundary → read** | yes | own written state | above chance | **the Stage 4W condition** |
| **state replacement** | yes | `W(X'_t)` after boundary | follows `S'` | the decision is driven by state, not by `X_t` |

The write-only arm is the negative control that makes the positive result
interpretable. The state-replacement arm is the one that separates causation
from correlation.

## 3. Structure

```
PRE-BOUNDARY
                    │
                    ▼
                  X_t
                    │
                    ▼
             WRITE_θ(X_t)
                    │
                    ▼
                S_{t+1}
                    │
════════════════════╪════════════════════
          TEMPORAL BOUNDARY
════════════════════╪════════════════════
                    │
                    ▼
                 Q_{t+1}
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
     S_{t+1}                S'_{t+1}
     natural              intervention
        │                       │
        └───────────┬───────────┘
                    ▼
                  READ_θ
                    │
                    ▼
                    Y
```

Compare `A_natural` against `A_no-state`, `A_corrupt-state`, and `A_sham`.

## 4. Falsification criteria — stated before implementation

Stage 4W is falsified, and no claim promoted, if **any** of the following holds.

**G1 — write representability fails.** The ideal writer and reader, wired
through the actual state mechanism, do not separate gold from the strongest
distractor. Then the mechanism cannot express the solution and no learning
result is interpretable in either direction. *(This is §34 applied twice — once
per direction of flow.)*

**G2 — boundary leaks.** With `S_{t+1}` ablated, post-boundary accuracy remains
materially above chance. A leakage channel exists and the run is **inadmissible**.
Hard stop.

**G3 — no causal effect.** `do(S_{t+1} \leftarrow S'_{t+1})` does not move the
decision in the predicted direction at the pre-registered threshold. The state
correlates with success but does not drive it. Threshold: `0.30`, carried forward
from Stage 3b's F2 so the two interventions are comparable.

**G4 — stateless explanation.** The write-only arm, with the written state
destroyed, performs materially above chance. The information crossed the boundary
through something other than state. Hard stop.

**G5 — weights carry the answer.** Accuracy depends on the written content being
drawn from a fixed, small set rather than freshly randomised per episode. The
parameter-memorisation loophole is open. *(Prevented by construction per §1; if
it is found anyway, the environment is invalid.)*

**G6 — read-only does not work.** The read-only arm cannot retrieve a supplied
correct state. Then the failure is in retrieval, not in write-boundary-read, and
the experiment is measuring a broken reader rather than a boundary.

## 5. The methodological pair

Stage 3b taught:

> Do not interpret a negative learning result until representability has been
> established.

Stage 4W needs a second rule, which is registered here:

> **Do not interpret successful post-boundary behavior as state persistence until
> alternative information channels have been experimentally eliminated.**

Together:

\[
\boxed{\text{Representability} \;+\; \text{Boundary integrity}}
\]

Without the first, a negative result can be a broken mechanism.
Without the second, a positive result can be a shortcut.

Stage 3b was vulnerable to the first failure only, because it had a negative
result to explain. Stage 4W is vulnerable to the second, because it expects a
positive one. The controls are therefore weighted oppositely: §34 protects
against concluding the substrate is incapable, and Layer 2 protects against
concluding it is responsible.

## 6. Constraints — binding

Carried forward from Amendment 2 §4 and the program's standing constraints:

1. **No architectural change.** No recurrence, no memory augmented beyond the
   existing `PersistentState`, no attention over the boundary, no dual paths.
2. **No new components beyond the writer.** No verifier (that is the other
   Stage 4), no critic, no reward shaping, no contrastive loss, no curriculum.
3. **No environment redesign of the Stage 3b machinery.** Reuse the existing
   exact-`T` construction and the key-relative feature basis where possible; the
   Stage 3b invariants must continue to hold.
4. **No additional tuning.** The learning rates are those already tested in
   Stage 3b. No search over boundary length, batch size, or stream length.
5. **FRESH randomisation per episode.** The written content must not be
   predictable from the parameters (§1).
6. **Retractions stand.** The Stage 3b invalidation chain is not reopened, and
   no superseded number is cited as evidence.

## 7. What is and is not claimed on a pass

**Would be claimed (narrow scope):**

- Episode-specific information can be written into persistent state.
- It survives an information boundary across which the original observation is
  unavailable.
- It causally controls a later decision, established by replacement rather than
  correlation.

**Would NOT be claimed, even on a pass:**

- long-horizon memory beyond one boundary;
- multi-step persistent computation;
- state composition or multi-fact state;
- verifier-guided repair (a separate experiment);
- continual learning, or resistance to drift;
- generalization to unseen state structures;
- scaling beyond the tested key space;
- the full PNDS decision loop, which still lacks a verifier.

Stage 4W does not make the system a memory. It makes state a **carrier**, for
one fact, across one boundary. That is enough to be worth establishing, and it
is small enough to establish cleanly.

## 8. Implementation order

1. **This document committed first.** No Stage 4W code exists before it.
2. The **write representability gate** (Layer 1) is implemented and must pass —
   and must be shown to fail on a defective writer — before any training.
3. The **boundary integrity gate** (Layer 2) is implemented and must hold.
4. Only then is the learned writer trained and the four arms of Layer 4 run.
5. Results recorded as `STAGE_4W_RESULT.md`, with the §34 gate reproduced in the
   record rather than referenced.

If Layer 1 fails, the run stops and the failure is recorded. A gate failure is a
result, not an obstacle.
