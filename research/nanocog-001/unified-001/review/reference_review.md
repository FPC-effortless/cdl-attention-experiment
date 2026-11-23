# COG-UNIFIED-001 independent semantic-reference review

Reviewer: `unified_mechanisms`, a separate agent session sharing the workspace. This is an adversarial source review and constructed-input execution, not independent scientific replication.

## Scope

Inspected `reference/semantic_kernel.py`, `reference/check_contracts.py`, and `ARCHITECTURE.md`. Independent tests are in `review/reference_independent.py`. No historical model code, checkpoint, neural fit, benchmark dataset, protected evaluation or serving measurement was used.

The review concentrates on invariant boundaries that the author's original 48 checks could miss: divergent immutable forks, content-versus-counter identity, transitive COPY dependencies, exact absence, malformed transaction observations, cache parameter validation, and policy propagation.

## Finding 1 — revision counters do not identify a fork's contents

**Severity: correctness defect. Repaired in the reviewed candidate.**

From one valid state A=red at revision 1, two legal commits create A=green at revision 2 and A=blue at revision 2. Both retain the same declared epochs. The previous rule compared only the per-key numeric revision.

Consequences of that rule:

- A pure-read cache created on the green branch could be reused on the blue branch and return green.
- A COPY transaction prepared on the green branch could commit on the blue branch and write the wrong snapshot value.
- A predicate-dependent write could miss the conflict even though its source fact differed.

The defect did not require a forged cache, invalid store or hostile Python object. It followed from ordinary immutable branching.

The author replaced numeric-only dependencies with the full immutable Cell observation (revision, payload, schema, evidence and producer), or exact absence, plus epochs. The independent fork-cache, transitive-COPY and predicate-read tests pass with this change. The architecture now explicitly rejects bare counters as sufficient cross-fork identities.

**Evidence handling:** the old complete source was inspected but not executed by this reviewer before repair. The suite preserves an executable counterexample to the exact reconstructed old revision-only predicate, clearly labelled as such. It must not be reported as an old-source replay.

A production optimization may replace full-cell comparison only under an explicit integrity/incarnation contract. A compact hash or UUID is not automatically collision-free, authentic or globally unique. Its construction and comparison costs belong in the complete cost model.

## Finding 2 — malformed immutable state and cache-input validation

**Severity: boundary correctness and bounded-state specification. Repaired in the inspected candidate.**

The first source accepted a tuple containing mutable entry rows, did not validate Epochs and their field types/bounds, and evaluated the cache-hit path before validating the hop cap. A boolean or floating value equal to a cached integer cap could therefore bypass resolve's strict integer validation.

The inspected repairs:

- Require immutable two-element entry tuples.
- Require a valid Epochs object with bounded string fields and an exact integer authority domain.
- Bound store capacity and schema metadata.
- Validate key and integer hop cap before testing a cache hit.
- Guard transaction entry shapes before tuple unpacking.
- Cap declared reads and transaction writes, including the number of iterable elements consumed.

The first executed independent suite passed 26 checks after these repairs. Its receipt is `reference_independent_first.json`. It is not a pre-repair failure receipt.

The count cap assumes each supplied iterator step terminates. It is not a wall-time bound on arbitrary Python callbacks. Likewise, type validation does not make Python dataclasses secure capabilities against deliberate process-memory mutation.

## Finding 3 — bool/float equality can bypass transaction domains

**Severity: malformed-input acceptance. Repaired and independently rechecked.**

An expanded suite found three actual failures:

1. `Cell(revision=True, ...)` compared equal to the valid revision-1 observation.
2. `Cell(revision=1.0, ...)` also compared equal.
3. `Epochs(authority=True)` compared equal to a valid authority-1 epoch.

The store constructor rejected these domains, but commit initially checked only observation class and equality. Imported transaction values therefore bypassed those domain rules.

The structured failure receipt `reference_independent_epoch_probe.json` pins source, script and architecture SHA-256 identities and records **32 checks: 29 pass, 3 fail**. These are actual executed findings.

The final source validates the complete expected Cell and epoch domains before equality. All three rejecting regression cases pass, preserving the original store. A type-correct comparison is still not proof that a caller truthfully declared every semantic dependency.

## Finding 4 — protected-record policy must reach the one-step interface

The initial semantic_step wrapper had no protected argument and called commit without one. Direct commit users could supply a protected set, but the advertised processing-loop wrapper could not.

The final semantic_step interface forwards an explicit protected set to commit. Its independent regression passes. A loop-level preservation assertion remains conditional on receiving and enforcing the correct policy set. Protection of a LINK cell's bytes is intentionally weaker than protection of the answer obtained through its source dependency.

## Architecture and claim corrections requested and observed

- Carry every transitive resolved dependency into COPY and pure-read caches, including missing terminal records.
- Bind a cache to its hop cap and all interpretation/session epochs.
- Keep cached pure typed resolution separate from caching neural outputs that depend on local state, broker state and randomness.
- Treat single-process immutable commit as an abstract transition; a real shared store must atomically publish the compared-and-updated snapshot.
- State the new revision policy explicitly. The inherited typed prototype increments only on material changes; the unified reference increments every accepted write, including identical SET and repeated DELETE. DELETE on an absent key creates a tombstone identity. This affects lifetime capacity and revision cost.
- Replace 'existing admitted modern-attention roster' with the proposed roster after each comparator is admitted; PR90 has no admitted modern comparison result.
- Replace 'prove TRAIN adequacy' with empirical TRAIN/CAL demonstration under frozen criteria.
- A bounded active store does not bound total process memory. Cached answers, compiled procedures, retained snapshots, forks and sessions need explicit capacities or must be charged as growing state.
- Caller policy, epoch meaning, dependency disclosure and restore provenance remain assumptions. The reference neither infers semantic truth nor authenticates an untrusted externally supplied cache.

## What the independent checks establish

On the specifically executed finite constructed cases, the repaired reference distinguishes snapshot COPY from live LINK, rejects conflicting source observations, retains absence dependencies, rejects stale schemas, enforces pre-write reads, avoids partial writes on capacity/revision/policy failure, and does not conflate empty strings with missing values.

The suite additionally preserves counterexamples showing that raw LINK-cell protection does not imply linked-answer preservation and that revision counters alone are insufficient across branches.

## What this review does not establish

- A Python-to-Lean refinement proof or exhaustive implementation correctness.
- Correct entity/role inference, trained binding, successful learning or natural-language generalization.
- Causal honesty of a neural caller or completeness of its declared read set.
- Distributed serializability, crash consistency, durable persistence or external authority.
- Unbounded recall in bounded memory.
- An efficient production index; the reference uses linear tuple lookup and functional snapshot reconstruction.
- Safe caches accepted from untrusted callers; the ReadCache object is trusted internal data.
- A serving speedup or acceptable language quality.

## Final execution status

**ACCEPT_SCOPED_SEMANTIC_REFERENCE_WITH_LIMITATIONS.** All 34 final independent constructed checks pass. The three failing malformed-transaction checks are repaired; the valid-fork counterexamples are correctly refused under full-cell observation checks. Protected-policy forwarding and unconditional hop-cap validation also pass. Source inspection confirms the requested ADR, memory-ownership, empirical-adequacy and comparator-admission wording changes.

- Final semantic kernel SHA-256: `5ef3aeddf9112056c18fba90a35cbd0a69aba782ce131445f054223a339d0776`
- Independent script SHA-256: `4483fec373ba7aa81772509b3915933bd1436079aa67c7422215556709340972`
- Architecture SHA-256 at execution: `a8af3f9cf48c5302e1b1e143fd286d0c47a311de15aa03941c4fe787f82059c6`
- Finished UTC: `2026-11-23T05:16:26Z`
- Final receipt: `reference_independent_final.json`
- Preserved actual failing receipt: `reference_independent_epoch_probe.json` (29/32 pass; three failures)
- Earlier independent receipt: `reference_independent_first.json` (26/26 pass before expanding the suite)

These 34 checks are separate from the author's 48 constructed checks. Their count is not a statistical sample size, universal proof, model qualification or benchmark result. No remaining concrete blocker was found within this review's source and input scope. The explicit limitations above remain. Optional testing stopped after the identified defects were repaired.

