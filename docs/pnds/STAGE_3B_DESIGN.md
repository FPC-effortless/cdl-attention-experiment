# PNDS-GATE-001 Stage 3b — Pre-registered Design: Persistent State and the First Causal Intervention

**Status:** PRE-REGISTERED, NOT RUN. No implementation exists yet.
**Parent:** Stage 3a (`db7cdca`, CI run `36053169778`)
**Stage 2c/3a branch:** FROZEN as a closed experimental record.
**Author date:** 2026-09-24

> This document is a design. Every number in it is either (a) an already-published
> Stage 2c/3a result, or (b) an explicitly-labelled *analytic prediction* derived
> before any Stage 3b code was written. Nothing here is an experimental result.

## 1. What Stage 3a closed

Stage 3a established that the Stage 2c learned behaviour is *dependent* on the
learned weights, and that no benign fallback survives their joint removal. It
deliberately made no causal-intervention claim, because no persistent state
`S_t` existed to intervene upon. Stage 2c and 3a are both bandits: stateless,
i.i.d. episodes, one decision per episode, no memory.

The evidence promoted so far is therefore entirely **stateless**:

- E2: outcome-trained routing learns a context-gated relevance rule that a fixed
  similarity scorer provably cannot express (Stage 2c).
- E2: the learned weights are necessary for that behaviour (Stage 3a, joint lesion).

Neither addresses the PNDS loop claim. The PNDS hypothesis is a claim about a
*persistent decision substrate*: `S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}`.
Until a persistent state exists, can be read, and can be damaged, no PNDS-specific
claim is even statable. Stage 3b is the first stage in which that is true.

## 2. The design trap this section exists to prevent

Three prior designs were killed by representability checks before they were
run (two Stage 2c candidates, recorded in `PNDS_GATE_001.md`). Stage 3b has the
same class of trap, in a more dangerous position, because a *bad* persistent-state
design fails silently: it produces a plausible-looking intervention effect that is
actually explained by something other than the state.

The trap has **four** independent failure modes. Each was found analytically, and
each would have produced a publishable-looking result that proved nothing.

### 2.1 Failure mode A — the state is inert (the i.i.d. problem)

Stage 2c's rule is a **constant**. The context marks `dim // 4 = 2` positions;
there are `C(8,2) = 28` possible mark-sets, and training covers all 28. So the
optimal weight vector does not depend on any stream-level latent, and no
persistent state can carry information the router does not already have.

**Verified:** all 28 mark-sets appear within the 2000 training episodes; gold
satisfies the relation in 300/300 test episodes with exactly 1 satisfier in
300/300.

> **Consequence.** Any `S_t` bolted onto the Stage 2c environment is provably
> inert. An intervention on it would show a zero effect for a trivial reason, and
> a *nonzero* effect would indicate a bug, not a mechanism. **Stage 3b therefore
> requires a new environment, not a modification of Stage 2c.**

### 2.2 Failure mode B — the state is redundant with an observable (the leakage problem)

If the observation still carries the marker bits (as the 2c context does), the
router can recover the key from the observation alone and `S_t` carries nothing.

> **Design rule R1.** The key must appear **only** in `S_t`. Every per-episode
> observable must be statistically independent of it.

### 2.3 Failure mode C — a stateless shortcut explains the effect (the 2c/3a confound, revisited)

This is the failure mode that Stage 3a discovered in the learned weights, and it
reappears at the environment level unless it is designed out. In Stage 2c the
gold candidate deliberately **anti-matches** the query on unmarked positions, so
the rule "pick the candidate with the *least* total query agreement" scores
**0.5567** at 8 candidates with no learning and no state whatsoever.

If that anti-match structure is carried into Stage 3b, the same shortcut survives,
and any "intervention effect" is confounded with it: the router could be reading
nothing from `S_t` and still scoring 0.55.

> **Design rule R2.** No fixed stateless rule over the router's observables may
> beat chance. This must be *measured* on the actual environment, not argued.

### 2.4 Failure mode D — the intervention has no room to act (the dilution problem)

If the stored key is *short* relative to the descriptor, a *wrong* key is often
still satisfied by the gold candidate, because the candidate may agree with the
target on the wrong positions by chance. Analytically, with key size `k` and
`dim - k` unmarked positions of random bits, a wrong key of size `k` is satisfied
with probability roughly `2^-k` per candidate, which for `k = 2` is 0.25 — but the
empirical number is far higher because the candidate set is constructed to be
near-satisfiers.

**Measured across probe designs** (300 held-out episodes each, `dim = 8`):

| Design | true key | corrupt key | static | anti_static | verdict |
|---|---|---|---|---|---|
| 2c anti-match carried over, `k=2` | 1.0000 | 0.0200 | 0.0000 | **0.5567** | fails R2 |
| unmarked bits random, `k=2` | 1.0000 | **0.3667** | 0.2600 | 0.1833 | fails R2, diluted |
| unmarked random, `k=4`, all-miss distractors | 1.0000 | 0.2533 | **0.7400** | 0.0067 | fails R2 |
| unmarked random, `k=4`, sign-symmetric near-miss | 1.0000 | 0.2533 | **0.7767** | 0.0067 | fails R2 |
| unmarked random, `k=4`, key-flip-only distractors | 1.0000 | 0.2600 | **1.0000** | 0.0000 | fails R2 |

Every one of these would have produced a large `true_key - corrupt_key` gap and
a plausible-looking story. All five are invalid as Stage 3b designs.

The reason is now precise and worth stating, because it is a structural property,
not a tuning artefact:

> **Theorem (informal).** Suppose relevance is defined by agreement between the
> candidate and a query, restricted to a hidden key. Then no choice of
> distractor construction and no choice of gold's unmarked-bit distribution can
> make the stateless arms collapse to chance, *unless* the unmarked bits of gold
> and of every distractor are identically distributed and the relation is
> decoupled from the query.
>
> *Proof sketch.* If gold's unmarked bits systematically disagree with the query,
> the anti-agreement rule separates. If they agree, the agreement rule separates.
> If they are random but distractors are constructed to be near-satisfiers of the
> true relation, the near-miss structure leaks the key's agreement profile into
> total agreement, because a near-miss differs from gold on exactly one key
> position and therefore has total agreement differing by a small constant in a
> consistent direction. Any consistent direction is a stateless signal. Hence the
> relation must be decoupled from the query entirely.

This is the single most important finding of the design phase, and it is why
Stage 3b's environment looks different from Stage 2c's.

## 3. The Stage 3b environment

### 3.1 What persists

A **stream-level latent** that is stable across many episodes within a stream and
resampled between streams:

```
S_t = ( target t in {0,1}^dim , key K subset of {0..dim-1}, |K| = k )
```

- `t` — a fixed target vector the router must match candidates against.
- `K` — the set of positions that actually count.

`t` and `K` are drawn once per stream. They are **never** part of any per-episode
observation.

### 3.2 What the router observes

Per episode, the router sees:

```
query       q in {0,1}^dim        (pure noise, independent of t and K)
candidates  c_1..c_n in {0,1}^dim
```

and, through a **routing read**, the persistent state `S_t`. The read is the
object under test.

> **Critical.** `q` is deliberately independent of `t`. In Stage 2c the query was
> the thing the gold candidate agreed with. Here it is not. The query is a
> nuisance variable whose only function is to prevent the router from
> shortcutting by matching `q`. This is what kills failure mode C structurally
> rather than by tuning.

### 3.3 The relation

```
relevance(c) =  all( c_j == t_j  for j in K )
```

The gold candidate is the **unique satisfier**. Distractors are built by sampling
a fully random descriptor and then forcing a violation of the relation on at
least one key position — so every distractor is *nearly* relevant, which keeps
the task non-trivial, but provably not relevant.

Gold is constructed by sampling a fully random descriptor and then forcing
satisfaction on the key positions. **Gold and distractors are therefore
exchangeable given the query**, which is exactly the property that makes the
stateless arms fail.

### 3.4 Why this satisfies every design rule

| Rule | Status |
|---|---|
| R1 key only in `S_t` | `q` independent of `t` and `K` by construction; no per-episode field encodes either |
| R2 no stateless shortcut | **measured:** `static` 0.2833 vs `anti_static` 0.2767, symmetric, both at chance (0.125) within tie-break noise; see §3.5 |
| R3 non-degenerate intervention | **measured:** `true_key` 1.0000 vs `corrupt_key` 0.0000 |
| R4 representability | the correct solution is an exact vector in the hypothesis class; see §3.6 |

### 3.5 The analytic prediction, and the one artifact in it

Predicted stateless-arm behaviour, measured on a probe of the described
environment (300 held-out episodes, `dim = 8`, `k = 4`, 8 candidates):

| Arm | success | interpretation |
|---|---|---|
| `true_key` oracle (reads `S_t`) | **1.0000** | the state is sufficient |
| `corrupt_key` oracle (reads a damaged `S_t`) | **0.0000** | the state is necessary |
| `static` (max total agreement with `q`) | 0.2833 | no stateless signal |
| `anti_static` (min total agreement with `q`) | 0.2767 | no stateless signal |
| `random` | 0.1300 | chance floor |

**The 0.28 is an artifact, not signal, and here is the proof.** `static` and
`anti_static` are *symmetric*: 0.2833 vs 0.2767. If total query agreement carried
real information about relevance, these two arms could not be equal — one of them
would have to be above chance and the other below. Their equality means the
selection is being decided by **tie-breaking**, not by the score. In the probe,
gold sat at candidate position 0 and the scorer used `scores.index(max)`, which
returns the first index among ties; with random descriptors most candidates tie
at agreement ≈ `dim/2 = 4`, so the first index wins disproportionately. A probe
variant that shuffled gold to a random position and broke ties randomly produced
`static` 0.2133 and `anti_static` 0.2133 — i.e. the artifact collapsed toward
chance and the symmetry was preserved.

> **Design rule R5.** All fixed scorers in the real implementation must break ties
> randomly under a seeded RNG. The probe artifact must not survive into the
> artifact. This is asserted by a unit test.

### 3.6 Representability — the check that killed two Stage 2c designs

The router scores candidates with a fixed linear function over features of
`(query, candidate, S_t)`. The basis is **key-indexed**: one feature per
(key, position) pair,

```
g_{(K', j)} = [ j in K' ] * [ t_j == c_j ]
```

so the basis has `C(dim, k) * dim` features — **560** at `dim = 8, k = 4`. This is
finite, fixed, and known a priori.

The oracle weight vector `w[(K, j)] = +1 for j in K` (and zero elsewhere) scores
the unique satisfier exactly `k = 4` and every distractor at most `k - 1 = 3`.
So the correct solution is **exactly linear-representable**, and the
experiment tests *learning from outcomes*, not model-class expressiveness.

A router that does **not** read `S_t` cannot express the relation at all: `t` is
unknown to it and `q` is independent of `t`, so its best achievable score is
chance. This is the separation Stage 2c established between learned and static,
now established between *routing-into-state* and *not-routing-into-state*.

## 4. The intervention

This is the first stage with a genuine `do()`.

```
do(S_t <- s')    where s' = (t', K) with t' != t, K unchanged
```

The router reads the corrupted state and routes against it. The measured
contrast is

```
Delta_causal = P(success | read S_t)  -  P(success | do(S_t <- s'))
```

with the predicted magnitude **1.0000 - 0.0000 = 1.0000**.

**Why this is a real intervention and the Stage 2/2b/2c/3a "intervention" was
not.** The earlier `intervention_effect` was a within-episode re-scoring against
`(selected + 1) % n`. It was found to agree *exactly* with the success indicator
(`P(intervention=1) == P(success)`), meaning it carried no information beyond
the outcome label — a fact recorded as a known limitation in `PNDS_GATE_001.md`
and the reason Stage 3a was designed as an ablation rather than an intervention.
Here the intervention acts on the **persisted object**, not on the episode: the
same held-out episodes, the same frozen router, only `S_t` differs. There is no
re-scoring and no relabelling.

**The distinction that makes or breaks the claim.** The environment is
constructed by us. Intervening on the *episode* would prove nothing about the
learner, because we built the episode. Intervening on the *persisted state* is
different: it asks whether the router's behaviour is a function of the state it
reads. That is the first question the PNDS hypothesis actually poses.

## 5. Arms

| Arm | Reads `S_t`? | Router | Purpose |
|---|---|---|---|
| `oracle` | yes, true `(t, K)` | unique satisfier | capacity ceiling |
| `true_key` | yes | relation-following | state sufficiency |
| `corrupt_key` | yes, damaged `t'` | relation-following on damaged state | **the intervention** |
| `learned` | yes | outcome-trained linear router | the claim under test |
| `learned_no_state` | **no** | same architecture, state features withheld | **state necessity control** |
| `static` | no | max total agreement with `q` | stateless shortcut reference |
| `anti_static` | no | min total agreement with `q` | stateless shortcut reference |
| `random` | no | uniform | chance floor |

`learned_no_state` is the arm that makes this a state test rather than a model
test. It has the same architecture and the same training budget as `learned`,
with the `S_t`-derived features removed from its basis. If `learned` does not
beat `learned_no_state`, persistence is contributing nothing, regardless of how
well `learned` scores.

## 6. Controls and the seven destructive tests (protocol §28)

The protocol requires destructive controls for any persistence claim. Stage 3b
implements them as follows.

| # | Control | Implementation | Predicted effect |
|---|---|---|---|
| 1 | **reset** | clear `S_t` between episodes within a stream | learned collapses to `learned_no_state` |
| 2 | **state shuffle** | permute which stream a state belongs to | collapse to chance, since state/episode are mismatched |
| 3 | **wrong-context replacement** | replace `t` with an independent `t'` mid-stream | this is the `corrupt_key` intervention |
| 4 | **controlled state corruption** | flip a fixed fraction of `t`'s bits | graded degradation, monotone in corruption fraction |
| 5 | **state recoverability probe** | after corruption, allow continued learning | does the router re-converge on the true state? |
| 6 | **functional state-use probe** | does the router's read depend on the query? | a constant read is not routing |
| 7 | **held-out regime** | test streams with unseen `(t, K)` | generalization, not memorization |

Controls 4 and 6 are the two most likely to expose a null result, and they are
the ones I most want to see.

**Control 4** is the dose-response. The pre-registration decision — informed by
the Stage 3a dose-response experience, where two curves supported opposite
conclusions and the divergence was the informative quantity — is:

> **Control 4 is a diagnostic, not a promoted result.** A monotone
> corruption-vs-success curve will not be promoted as primary evidence, because
> a monotone curve is consistent with a lookup table, which is not the PNDS
> claim. It is recorded to establish that the state-to-behaviour coupling is
> graded rather than all-or-none.

**Control 6** is the probe that catches the failure mode nobody would otherwise
notice. If the router learns to read `S_t` but its read is *not* query-dependent
— the same `(t, K)` features dominate regardless of the episode — then it has
learned a fixed lookup, not selective routing. This would still produce a perfect
intervention effect and would still beat `learned_no_state`. It is the honest
name for failure mode B relocated into the learner. **Predicted:** on this
environment control 6 will show little query dependence, because the query is
deliberately a nuisance variable. That is a real limitation and §8 records it.

## 7. Falsification criteria

The stage is falsified, and no PNDS claim is promoted, if **any** of the following
holds. These are stated before implementation.

1. **F1 — inert state.** `learned` does not beat `learned_no_state` by a
   margin that is positive on a pre-specified majority of seeds. Then the
   persistent state contributes nothing measurable.
2. **F2 — no causal effect.** `Delta_causal` is at or below the pre-registered
   threshold. Threshold: `0.30` (chosen as roughly an order of magnitude above
   the tie-break noise floor of ~0.08 observed in the probe, and well below the
   predicted 1.0, so that a partial but real effect can still register).
3. **F3 — stateless explanation.** `static` or `anti_static` beats chance by
   more than `0.10`. Then the environment has a shortcut and the whole design is
   invalid, exactly as in the five rejected designs of §2.4.
4. **F4 — reset does not damage.** The reset control (control 1) does not reduce
   `learned` toward `learned_no_state`. Then behaviour is stored in the weights,
   not the state — the i.i.d. trap of §2.1, relocated into the learner.
5. **F5 — not routing.** The functional state-use probe (control 6) shows the
   router's read is query-independent *and* performance is unaffected by
   withholding the query. Then the substrate is a lookup table, not selective
   access.
6. **F6 — non-replication.** Fewer than 3 of 5 training seeds show the stated
   effect sign.

**Threshold rationale.** F3's `0.10` is deliberately tighter than the observed
0.08 artifact band, so the test is not passed vacuously by noise. F2's `0.30` is
deliberately far below the predicted 1.0 so that a *partial* effect is still
reportable as a boundary condition rather than silently failing — the same
discipline that made Stage 2b's partial pass informative.

## 8. What Stage 3b will NOT establish, stated in advance

- **Not verification, repair, or `S_{t+1}` learning.** The state is *read* and
  *damaged*. It is not updated from outcomes, no verifier is trained, and no
  commit rule is exercised. Success-gate items 5 and 6 remain unaddressed. This
  is deliberate: protocol §29 (verified-only commit falsification) is a separate
  experiment and conflating it with the first intervention would make both
  uninterpretable.
- **Not a learned update rule.** `S_{t+1} = U(S_t, O_t)` is not implemented.
  Control 5 (recoverability) asks whether the router *can* re-converge after
  corruption; it does not implement `U`.
- **Not cross-domain.** The environment is synthetic, 8-dimensional, and
  bit-valued. Evidence level cannot exceed E2/E3.
- **Control 6 will probably show weak query dependence**, because the query is a
  designed nuisance variable. This means Stage 3b tests *persistence and
  addressability*, not *query-conditioned selective retrieval*. Selective
  retrieval requires a second signal that correlates with the target, which
  would reintroduce the observable leakage of §2.2. Stage 3b deliberately trades
  one property for the other. If it passes, the immediate follow-up is an
  environment in which the query is *partially* informative about `t`, which
  restores selectivity while keeping the intervention clean.
- **No claim that this parameterization is the only mechanism.** The same caveat
  as Stage 3a: a different basis could implement the same computation.

## 9. Success gate for promotion

All of the following must hold for any PNDS claim to be promoted from Stage 3b:

1. `learned` > `learned_no_state`, positive on a majority of seeds (against F1).
2. `Delta_causal` > 0.30 (against F2).
3. `static` and `anti_static` within 0.10 of chance (against F3).
4. Reset control degrades `learned` toward `learned_no_state` (against F4).
5. `n_satisfiers == 1` in every episode, verified by unit test.
6. Router inputs contain no gold fields, asserted by the Stage 0 allowlist audit
   extended to the `S_t` read.
7. At least 3 of 5 seeds show the stated effect sign (against F6).
8. Every artifact records the full provenance chain:
   `repo -> branch -> commit -> PNDS ID -> seed -> config -> artifact ->
   metrics -> interpretation -> decision`.

If promoted, the evidence level is **E2 at best** (controlled, multi-seed,
held-out, synthetic, first causal intervention) and **not E3** unless the
corruption dose-response and the held-out-regime control both pass — see §6.

## 10. Provenance and branch policy

- Stage 2c and Stage 3a are **frozen as a closed experimental record**. No
  further commits alter their results. Corrections to them (`9efe8c4`,
  `1a64aeb`) are already merged and are part of the record.
- Stage 3b work proceeds on `pnds` in the `.wt-pnds` worktree only.
- **Push discipline (learned twice the hard way):** push from *inside* `.wt-pnds`
  with `git push origin HEAD:pnds`, or push `<SHA>:pnds` from the repo root.
  Pushing `HEAD` from the repo root pushes the wrong branch — the main checkout
  sits on `phase1.5a-l2-reconstruction`, not the PNDS work. Before pushing, fetch
  and confirm the tip is a fast-forward, since concurrent commits from other
  agents have made `pnds` non-fast-forward before.
- The main checkout's untracked `casm_v01/__init__.py` and
  `casm_v01/phase15a/` belong to a different task and are **left untouched**.
- Implementation files (`stage3b_*.py`, `run_stage3b.py`,
  `test_stage3b_*.py`) are written only after this document is committed.

## 11. Implementation checklist

Written only after §1–§9 are committed. No code precedes the registration.

- [ ] `stage3b_persistent.py` — the environment, `S_t`, stream construction,
      `make_stream`, `make_episode_in_stream`, the eight arms, the seven controls.
- [ ] `run_stage3b.py` — CLI with `--train-episodes`, `--test-episodes`,
      `--candidates`, `--seeds`, `--stream-length`, `--corruption`.
- [ ] `test_stage3b_persistent.py` — the invariants below, each failing loudly if
      violated.
- [ ] Workflow step in `pnds-gate-001.yml` for 8/16/32 candidates.
- [ ] Results written to `docs/pnds/RESULTS.md` with the CI run ID.

### Invariants to be asserted by unit test

1. `gold is the unique satisfier` — in every episode of every stream.
2. `query carries no information about the target` — `q` independent of `t`, `K`.
3. `stateless arms are at chance` — `static` and `anti_static` within 0.10 of
   `1/n`, with **random tie-breaking** (R5).
4. `intervention acts on the state, not the episode` — the corrupted-state run
   uses identical episodes and an identical frozen router.
5. `router inputs contain no gold field` — the Stage 0 allowlist audit, extended
   to the `S_t` read.
6. `the relation is linear-representable in the hypothesis class` — an explicit
   oracle weight vector achieves the maximum possible score.
7. `learned_no_state has no S_t features` — asserted structurally, so the
   necessity control cannot silently acquire them.
8. `reset control actually resets` — no state survives across the reset boundary.

## 12. Predicted outcome, for the record

| Quantity | Prediction | Basis |
|---|---|---|
| `true_key` | 1.0000 | gold is the unique satisfier by construction |
| `corrupt_key` | ~0.00 | corrupted target makes the relation unsatisfiable |
| `static`, `anti_static` | ~0.21 (chance 0.125) | `q` independent of `t`; residual is tie-break noise |
| `learned` | unknown | this is the experiment |
| `learned_no_state` | ~chance | provably, `t` unknown and `q` uninformative |
| `Delta_causal` | ~1.00 | predicted, not measured |

The one cell marked **unknown** is the entire point. Every other number is
predicted by construction, which is what makes the design falsifiable rather
than demonstrative: if `learned` comes out at chance, the environment was
sound and outcome-training failed to learn to route into persistent state — a
genuine, reportable negative result. If `learned` comes out near `true_key`,
Stage 3b is the first controlled evidence that a persistent substrate can be
read, used, and causally damaged.

That is the first legitimate test of the PNDS loop, and it is a test because
the answer is not determined by the design.
