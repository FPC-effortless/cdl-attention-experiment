# COG unified architecture: independent mathematical guarantee audit

**Reviewer scope:** mathematical analysis of the proposed composition, independent of the implementation author. This document contains proofs and counterexamples at the specification level. It does not certify learned behavior, numerical kernels, a Python implementation, training convergence, deployment, or scientific superiority. It records no new model experiment.

**Proposed state reviewed:** `S = (h, P, W, E, I, C, theta_epoch, tokenizer_epoch, policy_epoch, authority_epoch)`, where `h` is causal local neural state, `P` is a bounded broker/packet state, `W` is an exact versioned typed store, `E` is retained evidence/history, `I` is a derived index, and `C` is a dependency-certified derived cache. A learned raw-text binder proposes namespace/entity/relation keys and payloads. A transactional executor applies SET, snapshot COPY, live LINK, and DELETE. Reads are bounded and can return UNKNOWN. Exact history is allowed to grow; the fast working set is bounded. The proposed native candidate excludes token-to-token attention and borrowed SSM implementations; support-supervision findings are transferred as a training hypothesis rather than importing the attention-style binder that produced the historical gain.

## 1. Overall finding

The fusion can support substantial **conditional structural guarantees**, provided the executor, cache certificates and version rules implement the stated contracts. It cannot guarantee all desired capabilities simply by combining previously successful mechanisms. In particular, finite memory, bounded communication, perfect unrestricted recall, universal semantic binding, unrestricted continual weight learning without forgetting, and universal speed superiority cannot all be promised together.

The design should distinguish five proof obligations:

1. The abstract state transition has the intended laws.
2. The executable implementation refines that transition.
3. Learned proposals correspond to the intended events and entities.
4. Training obtains that proposal behavior on the required distribution.
5. The complete implementation reaches an acceptable quality–cost point on actual hardware.

Proving an earlier obligation does not discharge the later ones. A sound exact executor can reliably execute an incorrect learned proposal. An exact provenance chain can identify the source of an incorrect claim. A correct cache can faithfully reuse a wrong model answer.

## 2. Exact typed-state laws

Let a raw store be a finite map from keys to versioned cells. A key is a tuple with explicit namespace and authority scope, not merely an untyped embedding. A cell is `Value(v)`, `Link(k)`, or `Tombstone`. Missing keys also have a validation identity so that absence can participate in transactions.

Write `put(W,k,c)` for an exact replacement of the cell at `k`.

### T1. Raw frame law

For every `j != k`,

\[
\operatorname{put}(W,k,c)[j]=W[j].
\]

**Proof:** map update compares the queried key with `k`; at unequal keys it returns the original entry.

This is the correct no-collateral-write guarantee. It says nothing about queries whose live dependencies include `k`.

### T2. Overwrite and tombstone laws

At the updated key,

\[
\operatorname{put}(W,k,\operatorname{Value}(v))[k]
=\operatorname{Value}(v).
\]

A subsequent overwrite replaces the earlier current cell. DELETE places a tombstone with a new version; it must not silently expose a previous value through fallback retrieval. Retained evidence can still preserve the historical claim while the current-state query reports deletion.

These laws concern an admitted key/value proposal. They do not prove that the binder chose the correct key, value, event time or authority.

### T3. COPY and LINK must have different semantics

`COPY(a,b)` resolves `b` in the transaction's immutable snapshot, and writes the resulting value into `a`. A later write to `b` does not change the copied value at `a`.

`LINK(a,b)` stores a live reference. Querying `a` subsequently resolves `b`, subject to cycle/hop/deletion rules. A later write to `b` may intentionally change the answer at `a`.

**Counterexample to naive preservation:** start with `a = Link(b)`, `b = Value(red)`. SET `b` to blue leaves the raw cell `a` unchanged but changes `resolve(a)` from red to blue. Therefore the raw frame law cannot be advertised as answer preservation.

COPY must abort or return an explicit unresolved result if the source cannot be resolved; storing an arbitrary guessed value would invalidate the snapshot law.

### T4. Dependency-closed query preservation

Let `D_W(q)` include every raw cell, absence check, scope/authority fact, and model-policy dependency actually used to answer `q`. If transition `W -> W'` preserves every dependency and the decoder/routing interpretation remains fixed, then a deterministic answer computation returns the same result.

For pure bounded LINK resolution, a sufficient condition is that all cells and versions on the traversed path are unchanged. For a neural readout, the condition also includes its parameters, tokenizer, relevant local state and any cross-record normalization inputs.

**Proof idea:** induct on the deterministic evaluation trace. Each next operation sees the same input and dependency value, so it takes the same branch and returns the same result.

**Important boundary:** preservation of a record's value is insufficient when the query score softmax normalizes over additional records. Adding a competitor can change retrieval without changing the former winner. Such candidate membership and ordering must belong to the dependency contract, or the cache must be invalidated conservatively.

## 3. Transactions, concurrency and parallel execution

### T5. Atomic commit preserves a checked invariant only under a sound checker

Let `P(W)` be a decidable structural invariant. Suppose a transaction constructs `W_candidate`, checks `P(W_candidate)`, validates every snapshot dependency against current versions, and atomically publishes either all changes or none. Starting from a state satisfying `P`, every committed state satisfies `P`.

**Proof:** induction over committed transitions. Rejected proposals preserve the old state. Accepted proposals publish precisely the checked candidate on the validated snapshot.

The checker must be sound for `P`; checking a hash or schema alone does not establish arbitrary factual truth. External effects additionally require an execution/reconciliation contract: local state atomicity does not make a remote service transaction atomic.

### T6. Read validation is required, including reads of absence

Validation must cover transitive reads, writes, negative/absent reads, authority epoch and policy/model epochs relevant to admission. Merely validating write targets permits write skew.

**Write-skew counterexample:** initially `a = false`, `b = false`; the invariant is `not(a and b)`. Transaction A reads `b = false` and writes `a = true`. Transaction B reads `a = false` and writes `b = true`. Their write sets are disjoint. If both commit without validating their read versions, the invariant is violated.

An absent-key read is also a real dependency. If a transaction decides that a name is unused and another transaction creates it, the first transaction must detect the changed absence state. Tombstone and namespace epochs must not wrap silently.

### T7. Commutation of independent updates

Suppose deterministic transactions `f` and `g` have complete read sets `R_f, R_g` and write sets `W_f, W_g`, and their admission preconditions depend only on these reads. A sufficient condition for semantic-store commutation is

\[
W_f\cap(W_g\cup R_g)=\varnothing,
\qquad
W_g\cap(W_f\cup R_f)=\varnothing.
\]

Then `f(g(W))` and `g(f(W))` have equal semantic store contents, assuming each transaction's writes are functions of its unchanged read snapshot.

**Proof:** neither operation changes the other's read inputs or written cells. On the union of writes, each final value is its operation's fixed proposal; outside that union, both preserve the old value.

Live references require the read sets to contain their transitive dependencies. Shared counters, quotas, authority changes and index metadata can introduce extra conflicts.

This theorem does **not** promise identical ordered audit logs, commit timestamps, generated transaction IDs or receipt sequences. Those need a canonical serialization or a separately specified equivalence relation. Parallelism can preserve the semantic store while changing the history's presentation order.

## 4. Correct cache reuse

### T8. Dependency certificate validity

For a cached result `c = F(q,S)`, record the query identity and every dependency of `F`: transitive cell versions, absence versions, candidate-set/index revision when selection depends on it, authority/policy/model/tokenizer epochs, and any local-state input. If all recorded identities match and `F` is deterministic under that identity, reuse returns the same result as recomputation.

**Proof:** the evaluation-trace argument of T4 applies to the whole cached computation.

Omitting one dependency is unsound. Useful counterexamples include a changed LINK destination, a newly introduced competing retrieval candidate, deletion followed by key reuse, new tokenizer segmentation, and changed output-head weights. A query-text digest alone is insufficient.

Exact equality can be proved in a mathematical model. If an implementation substitutes finite hashes for exact identity, collision resistance becomes an additional implementation assumption. The proof should not treat arbitrary hash equality as logical equality of bytes.

### Typed payload compatibility after weight changes

Invalidating caches at a new model epoch prevents reuse of old **outputs**. It does not automatically preserve the meaning of old **latent memory payloads**. A new reader may interpret an unchanged vector differently.

Use a stable typed value interpretation, retained raw supporting evidence and producer revisions. Alternatively, make compatibility an explicit checked migration or refuse incompatible reads. Neither a producer tag nor retained raw text itself guarantees successful semantic migration.

## 5. Causality and parallel local computation

### T9. Prefix causality

Fix parameters, initial state and random stream. If every transition and output at time `t` depends only on the current state and tokens up to `t`, two streams with the same prefix have identical outputs through that prefix.

**Proof:** induction over prefix length.

Chunked execution must preserve the transition order for memory events. Randomness should be keyed consistently, or chunk-dependent random-number consumption can defeat bitwise parity even when the probability law is causal. Floating-point reassociation can also change numeric results; mathematical equality is not bitwise implementation equivalence.

### T10. Associative affine scan has a precise applicability condition

For

\[
h_t=A_t h_{t-1}+b_t,
\]

where each `(A_t,b_t)` is available from token-local inputs independently of earlier `h` and memory feedback, define

\[
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,A_2b_1+b_2).
\]

This composition is associative by matrix multiplication associativity and distributivity. A parallel prefix scan can recover every prefix transform.

State-dependent gates, retrieved-memory feedback and event decisions generally break the availability condition. Abstract function composition remains associative, but that observation does not give a bounded-size cheap scan representation.

Even valid scan algebra does not prove lower wall time. Generic matrix composition can cost more work than sequential matrix-vector updates. Diagonal or another closed structured form is required to preserve the intended work bound. Memory-write barriers and conflicting updates remain ordered unless T7 applies.

## 6. Stability does not guarantee durable memory or learnability

For an exogenous affine recurrence with `||A_t|| <= rho < 1`, equal later inputs imply

\[
\|h_{t+k}-\tilde h_{t+k}\|
\le \rho^k\|h_t-\tilde h_t\|.
\]

This gives perturbation stability. It also attenuates old distinguishing information and the corresponding linear gradient path. Long-term exact facts therefore require another channel, such as the exact store, and successful learned writes to that channel.

A sigmoid gate in `[0,1]` is not by itself a contraction proof when the gate depends on `h`. The Jacobian contains derivative-of-gate terms. A valid theorem must bound the full Jacobian, or constrain the coefficients to input-only dependence.

Neither nonzero gradients nor a large memory-read norm guarantees correct entity/value binding. Optimization, representation identifiability, data coverage and loss design remain separate obligations.

## 7. Three unavoidable limits

### N1. Finite state cannot guarantee exact unbounded recall

Suppose the total retained state has at most `B` bits and must answer the value of any of `n` arbitrary binary facts after reading them, with no access to an external retained record. There are `2^n` possible fact assignments and at most `2^B` states. For `n > B`, two different assignments share a state. They differ at some key; a query for that key has the same retained state in both cases but requires different answers. At least one answer must be wrong.

Randomization cannot supply a probability-one guarantee for both indistinguishable histories. An external exact archive avoids the premise by adding retained information; its storage and access cost must then be counted.

**Design consequence:** bounded fast state plus a growing archive is coherent. Fixed total bytes plus unrestricted exact recall is not. A capacity limit must cause explicit refusal, lossy eviction, or a narrower promise about retained facts.

### N2. A bounded candidate budget does not guarantee universal relevance

In an unstructured collection of `N` candidates whose relevance is known only after inspecting each candidate, any algorithm inspecting fewer than `N` candidates can miss the sole relevant item. Construct two collections that agree on all inspected candidates and differ only at an uninspected location; the algorithm receives the same observations but one collection requires returning that location.

An exact key index can escape this unstructured-query premise when the correct key is available and its contract is satisfied. A learned semantic router does not acquire that guarantee merely by returning `K` candidates.

**Design consequence:** bounded routing guarantees a resource cap. It cannot also guarantee universal retrieval completeness. The model needs an explicit UNKNOWN/escalation path, and its coverage must be measured.

### N3. Finite regression tests do not guarantee universal non-forgetting

For any finite regression set `T` and an unrestricted function class, choose an input `x*` outside `T`. Functions `f` and `g` can agree on every member of `T` while differing at `x*`. Passing the suite therefore proves preservation on `T`, not all future inputs.

Universal non-interference is possible only under extra structure: for example, a frozen protected function and an update confined to a disjoint computational path whose selector cannot activate for protected queries. The selector itself is part of the proof obligation. For a linear protected readout, a parameter update in the exact nullspace of all protected features suffices, but checking a finite sample's feature span is a guarantee only for that span.

**Design consequence:** exact record preservation can be guaranteed by the executor. Preservation of all learned skills under arbitrary weight updates cannot be claimed from bounded tests or low-rank edits alone.

## 8. Semantic binding and paraphrase guarantees

Role factorization, training-only support labels, preservation losses and readout repair are reasonable candidate inductive biases. They do not logically guarantee that a new entity/relation/value triple is parsed correctly.

An exact semantic-invariance theorem would require a specified meaning-preserving transformation set `G` and an encoder/readout satisfying

\[
F(gx)=F(x)\quad\text{for every }g\in G.
\]

For entity renamings, equivariance may be the correct law instead: renaming the input and stored keys must rename the answer consistently. If the law holds for every generator of a transformation group and the implementation composes those generators faithfully, it extends to their compositions. Training on examples of those transformations does not prove that law.

Factorized scores also need an injective or otherwise task-sufficient key representation. Two distinct entity–relation pairs that collide in the representation cannot be distinguished by a downstream function of that representation alone. Adding separate heads does not by itself prevent such collisions.

Ambiguous observations impose another limit: two worlds can produce identical observed text while requiring different correct actions. No model can guarantee the right action in both without additional evidence or an abstention/query mechanism.

The executor's theorem should therefore be stated as: **if the admitted proposal identifies the intended typed event, then execution implements that event's specified semantics**. It must not silently assume the desired learned binding result in a theorem advertised as proving binding.

## 9. Conditional end-to-end reliability

Let failure events include acquisition error, incorrect entity/relation binding, missing retrieval, wrong value decoding, stale admission, implementation divergence and unsound external verification. If a correct final answer is implied by the absence of these events, then

\[
\Pr(\text{incorrect final answer})
\le\Pr\!\left(\bigcup_i E_i\right)
\le\sum_i\Pr(E_i).
\]

The union bound requires no independence. It requires justified bounds for the actual operating distribution. Training or development error rates are not automatically such bounds, especially under adaptive routing, selective acceptance or distribution shift.

Alternatively, if each stage succeeds with probability at least `1-epsilon_i` conditional on all preceding stages succeeding, the chain rule gives success at least the product of those conditional lower bounds. This is not an independence assumption; the conditional bounds themselves must be established.

Two agreeing verifiers that use the same evidence or failure mechanism cannot be treated as independent confirmations. Provenance checks can expose shared origins but do not make distinct sources correct.

Refusal can protect a soundness objective while reducing availability. A system that never answers trivially avoids wrong answers; it does not satisfy a useful task-success guarantee. Soundness, coverage, latency and correction success must be evaluated jointly.

## 10. Resource and efficiency claims

Bounds on `K` candidates, `H` link hops, `P` publications and `U` writes can guarantee upper limits on those operation counts. They do not automatically bound the total cost unless candidate construction, index maintenance, payload sizes, validation and output generation are also bounded and charged.

A representative per-token cost decomposition is

\[
C=C_{encode}+C_{local}+C_{route}+C_{index}+C_{read}
+C_{bind}+C_{verify}+C_{write}+C_{decode}+C_{retry}.
\]

For a growing store of `N` records, an exact balanced-tree index may contribute logarithmic key operations, while a semantic scan may contribute linear work. Tensor maps may cost quadratically in hidden width, and a dense vocabulary projection costs proportionally to vocabulary size times width. Merely replacing the history-attention operation does not remove these costs.

Fast working memory can be bounded while exact history storage grows with accepted information. Append-only history and version metadata must be included in total memory and persistence cost. LINK dependency validation can be as expensive as traversing the dependency closure it certifies.

### Reuse and learned search

For reuse construction cost `C_build` and valid per-use savings `Delta > 0`, the simple break-even condition is `n * Delta > C_build`, with validation, invalidation and rebuilds included in the respective terms. A learned trigger can reduce expected cost only when its benefit estimates are sufficiently accurate for the workload.

Learned tie ordering can preserve an A* solver's correctness when it only reorders equal-priority eligible nodes and the underlying algorithm's admissibility, successor completeness and reopen assumptions hold. It does not guarantee fewer expansions or faster completion on every problem. The architecture should expose this as an optional constrained controller mechanism, not a mandatory per-token planning loop.

### No universal speed or quality theorem

The fused system introduces costs that some simpler tasks do not need. Hardware kernels, batching, precision, model size, target quality and workload determine the crossover. No algebraic architectural decomposition establishes superiority against all attention or recurrent implementations.

The user's training/inference preference can be represented by `J = C_train + 12*C_infer` for a declared unit of inference use. It is a chosen accounting objective, not an intrinsic theorem; units, request counts and the quality constraint must be explicit. Comparing `J` across unequal competence does not establish a successful replacement.

## 11. Recommended guarantee ledger

| Desired property | Strongest defensible status |
|---|---|
| Causal outputs | Provable for the causal abstract transition under fixed parameters/randomness and prefix-only inputs. |
| Exact SET/COPY/LINK/DELETE semantics | Provable for admitted well-typed operations, with explicit unresolved/cycle/capacity behavior. |
| Untouched raw cells remain unchanged | Provable frame law. |
| Unrelated answers remain unchanged | Conditional on complete dependency closure and unchanged interpretation/selection. |
| Transaction invariant preservation | Conditional on sound checking, full read validation and actual atomic publication. |
| Correct cache reuse | Conditional on complete dependency certificates and identity integrity. |
| Parallel equality | Provable for a specified independent operation class; does not imply bitwise logs or runtime speed. |
| Resource caps | Provable for counted bounded operations; full resource caps require bounded discovery/maintenance/fallback. |
| Exact arbitrary long-term recall in fixed total bytes | Impossible for unrestricted facts. |
| Correct binding for every unseen phrase | Not implied; requires a restricted semantic contract or empirical evidence. |
| No forgetting after arbitrary weight updates | Not implied; finite tests are insufficient. |
| Better quality and speed than strong LLM baselines | Empirical and presently unestablished for the fusion. |

## 12. Acceptance recommendation for the fused specification

Accept the direction as a **conditional architecture specification**, provided all of the following remain explicit:

- Hard exact state operations are separated from learned event proposals.
- COPY snapshots and LINK live dependencies are different types/semantics.
- Missing evidence, cycles, exhausted budgets, incompatible latent payloads and capacity exhaustion return explicit unresolved outcomes.
- Transactions validate read dependencies, including absence and authority epochs.
- Cache identity includes candidate-selection dependencies as well as retrieved records.
- Parallel proofs target semantic state equivalence and do not hide global metadata conflicts.
- The archive is allowed to grow, or the retention guarantee is narrowed to an explicitly bounded set.
- The affine scan applies only to the input-dependent local subpath; event feedback is not silently treated as scan-parallel.
- Learned plasticity and unfamiliar-language generalization remain unproven capabilities.
- The fusion retains individual and combined-system ablations and full-cost, matched-quality comparisons.

This is enough to make the design mathematically precise and falsifiable. It is not enough to announce that the complete attention replacement is guaranteed to work. The remaining learning, refinement and economic obligations are substantive research tasks, not consequences of the structural proofs.

## 13. Adversarial inspection of the concrete architecture draft

I inspected `ARCHITECTURE.md` after drafting the mathematical analysis above. The draft correctly preserves read-before-write behavior, separates typed execution from learned proposals, refuses universal finite-memory recall, distinguishes COPY from LINK, and states that no unified model has been trained. It also explicitly limits the affine scan law and avoids inheriting the numerical gains of the attention-style support-supervised binder.

### Material issue found: neural cache dependence on local state

The inspected section 3 makes next-token output depend on the current local feature `h` and old broker packets `P`. Section 6 initially enumerated query, stream, revision and record identities, without explicitly requiring `h/P` or an equivalent complete causal-context identity.

**Counterexample:** keep the query, record versions, stream ID and all model epochs fixed, but let the decoder output `red` when a local-state bit is 0 and `blue` when it is 1. An answer cache keyed only by the initially listed fields returns the same value in both states although recomputation differs. Stream identity alone does not identify the causal state.

**Required repair:** restrict the implemented cache guarantee to pure typed resolutions, or include an immutable full causal-context identity covering local recurrent and packet state for neural outputs. A complete prefix identity plus a deterministic fixed-parameter transition can suffice; an unverified compact hash introduces its own collision assumption. Charge validation cost. This finding was sent to the architecture author before finalization.

### Epoch coverage clarification

The transaction paragraph initially named the authority epoch explicitly. Admission must also validate policy/schema revisions when they affect permitted operations or payload interpretation, or explicitly perform current-policy/schema revalidation. Older model proposals can be admitted only under a stated compatibility policy; model-epoch tags alone do not establish semantic correctness. This clarification was sent to the author.

### Credit-assignment wording clarification

The phrase that a sufficient credit-assignment horizon is necessary for long-delay acquisition should not imply that long backpropagation is logically required. Local supervised writes, eligibility mechanisms or other validated learning rules can supply credit without a long differentiable path. The appropriate requirement is a training experience and credit mechanism capable of teaching the claimed long-delay behavior. Restoring a gradient is not sufficient for that behavior.

### Abstract Lean scope inspected

The first inspected `proofs/UnifiedLaws.lean` contained 33 theorem statements. The typed map-update, dependency, causality and counterexample statements have appropriately narrow hypotheses. In particular:

- `DependsOn` is a universal property of the reader; a captured runtime trace is not automatically proof that this property holds.
- `versionSound` is explicit. Equal version numbers from unrelated histories cannot be treated as proof of equal payloads.
- `gatedCommit` proves which Boolean branch is returned. Those lemmas do not prove actual transaction validation, atomicity, crash consistency or serializability.
- The affine laws use exact natural-number arithmetic; they do not certify float reassociation or an actual neural scan implementation.
- The finite regression and LINK counterexamples substantiate limits without claiming a complete information-theoretic formalization.
- The general counting argument in N1 is a written mathematical proof. The Lean file's collision obstruction and finite three-to-two-state example do not by themselves formalize the full arbitrary-bit cardinality theorem.

I requested additional snapshot-COPY and DELETE statements from the formalization owner if the final report intends to list those operation laws as machine checked. The final theorem count and compiler verification must be taken from the final proof owner's receipt, not this earlier inspection count. I did not modify the Lean file or independently recompile it in this review.

### Read-before-write consequence

In the proposed contract, `y_t` reads `W_t` and the accepted current proposal produces `W_(t+1)` afterward. Therefore the output's independence from the **current commit** is a valid structural property. It does not mean that the output ignores the current token: the current local feature can carry that token's content. An immediate same-event answer that relies exclusively on the committed store must wait for the next step, or use a separately declared read-after-write contract. The draft already states this distinction correctly.


## 14. Review closure and independent Lean kernel replay

The architecture author repaired all three reported issues. The final reread confirms: the executable cache is explicitly pure typed resolution; neural cache reuse requires complete local/packet/causal identity or a separate independence proof; commit validation includes authority, policy and schema/interpretation epochs; and credit-assignment wording permits supervised local writes and other validated mechanisms. These findings are closed at the architecture hash below.

I independently reread all 46 final theorem statements and the proof-scope document. Added COPY, SET, DELETE and live-LINK laws correctly describe the abstract interpreter. The new old-path dependency theorem proves local completeness specifically for the deterministic fuel-bounded LINK reader. The finite-run simulation theorem remains conditional on a one-step simulation relation and matching readouts; it does not establish those premises for the learned system. No further mathematical scope defect was found.

For independent executable verification, I copied the exact source to a fresh temporary directory, compiled it with Lean directly, generated a separate axiom-audit file from all theorem declarations, and invoked Lean on that file. I did not invoke the proof author's verification helper or reuse its compiled module. This is a separate agent process on the shared host, not a separate machine or external research-team replication.

### Exact replay receipt

```json
{
  "reviewer": "/root/unified_guarantee_review",
  "scope": "Independent fresh kernel compile and generated axiom audit on shared host; not independent research replication",
  "started_utc": "2026-11-23T05:10:21.068205+00:00",
  "finished_utc": "2026-11-23T05:10:22.741007+00:00",
  "compiler": "Lean (version 4.19.0, x86_64-unknown-linux-gnu, commit 6caaee842e94, Release)",
  "source_sha256": "2df339e39910803713498132af1c66185900acb5d7b71f1eafb5f9709a3118ff",
  "source_unchanged_during_replay": true,
  "theorem_count": 46,
  "placeholder_or_custom_axiom_tokens": [],
  "compile_returncode": 0,
  "compile_stdout": "",
  "compile_stderr": "",
  "audit_returncode": 0,
  "audit_statement_count": 46,
  "audit_names_match": true,
  "observed_standard_foundations": [
    "Classical.choice",
    "Quot.sound",
    "propext"
  ],
  "audit_stdout_sha256": "899a498d41537622de759cb3613207706250d479d0b7dba05407e501d51582c6",
  "audit_stderr": "",
  "architecture_sha256": "58ca24ed2494bcd77473929d7b2383da07d515d80777374bf255fe54adba04fb",
  "proof_scope_sha256": "9fbb6d292fb16940871b37d0c596da4dd58a87a0f55a05fe7173a4d9c084a947"
}
```

### Actual axiom-audit output

```text
'COGUnified.write_read_same' depends on axioms: [propext]
'COGUnified.write_read_other' depends on axioms: [propext]
'COGUnified.write_overwrite' depends on axioms: [propext, Quot.sound]
'COGUnified.distinct_writes_commute' depends on axioms: [propext, Quot.sound]
'COGUnified.rejected_commit_preserves' does not depend on any axioms
'COGUnified.accepted_commit_uses_proposal' does not depend on any axioms
'COGUnified.grounded_read_is_correct' does not depend on any axioms
'COGUnified.protected_cells_preserved' depends on axioms: [propext]
'COGUnified.written_version_strictly_increases' depends on axioms: [propext]
'COGUnified.version_monotone' depends on axioms: [propext]
'COGUnified.dependency_cache_sound' does not depend on any axioms
'COGUnified.cache_requires_same_reader_epoch' does not depend on any axioms
'COGUnified.local_dependency_cache_sound' does not depend on any axioms
'COGUnified.local_epoch_cache_sound' does not depend on any axioms
'COGUnified.dependency_preserving_write' depends on axioms: [propext]
'COGUnified.affine_composition_action' depends on axioms: [propext]
'COGUnified.affine_composition_associative' depends on axioms: [propext]
'COGUnified.affine_left_identity' depends on axioms: [propext]
'COGUnified.affine_right_identity' depends on axioms: [propext]
'COGUnified.run_append' does not depend on any axioms
'COGUnified.run_preserves_simulation' does not depend on any axioms
'COGUnified.simulated_run_has_same_readout' does not depend on any axioms
'COGUnified.prefix_output_independent_of_suffix' depends on axioms: [propext]
'COGUnified.edit_preserves_iff_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'COGUnified.protected_domain_preserved' depends on axioms: [propext]
'COGUnified.isolated_adapter_preserves_outside_route' depends on axioms: [propext]
'COGUnified.isolated_adapter_preserves_declared_domain' depends on axioms: [propext]
'COGUnified.finite_regression_passes' depends on axioms: [propext, Quot.sound]
'COGUnified.finite_regression_still_changes_output' depends on axioms: [propext]
'COGUnified.unchanged_read_path_preserves_resolution' depends on axioms: [propext]
'COGUnified.bounded_link_reader_locally_depends_on_path' depends on axioms: [propext]
'COGUnified.set_record_reads_value' depends on axioms: [propext]
'COGUnified.deleted_record_reads_none' depends on axioms: [propext]
'COGUnified.delete_preserves_other_cell' depends on axioms: [propext]
'COGUnified.live_link_dereferences_current_state' depends on axioms: [propext]
'COGUnified.snapshot_copy_materializes_value' depends on axioms: [propext]
'COGUnified.snapshot_copy_refuses_unresolved' depends on axioms: [propext]
'COGUnified.snapshot_value_survives_other_write' depends on axioms: [propext]
'COGUnified.link_cell_preserved' does not depend on any axioms
'COGUnified.live_link_answer_changed' does not depend on any axioms
'COGUnified.link_reads_bounded_by_fuel' depends on axioms: [propext, Quot.sound]
'COGUnified.disjoint_targets_still_interfere' does not depend on any axioms
'COGUnified.omitted_candidate_is_unread' depends on axioms: [propext]
'COGUnified.concrete_sparse_miss' depends on axioms: [propext]
'COGUnified.three_histories_two_states_collide' depends on axioms: [propext]
'COGUnified.encoding_collision_blocks_exact_recall' does not depend on any axioms
```

**Closure:** accept the repaired architecture as a conditional specification and the 46 statements as independently kernel-checked abstract mathematics. Preserve every learning, implementation-refinement, storage-engine and performance limitation described above. No neural model was imported, trained or benchmarked during this review.


## 15. Final mathematical review closure

I reread the final `MATHEMATICS.md` in full and the updated `ARCHITECTURE.md`, including the later independently discovered fork-identity repair and aggregate-memory limitations. This closure supersedes the earlier architecture hash in section 14; the Lean source remains unchanged from the independent successful kernel replay.

**Decision: accepted within its stated conditional scope. No blocking mathematical defect remains in these documents.**

The final text correctly distinguishes the 46 machine-checked abstract statements, written general mathematics, constructed Python checks and untested learned/deployed behavior. It does not claim an end-to-end trained model, empirical speed/quality improvement, a full Python refinement proof, or a Lean proof of database atomicity/serializability. The simulation theorem retains its one-step and readout premises; the counting lower bound remains separately identified as written mathematics. Historical component gains are not multiplied or inherited by the fusion.

The reference's complete immutable cell observations address the specified cross-fork counterexample where equal numeric revisions contain different values. The documents do not claim that this repair is itself a Lean refinement proof. Compact production identifiers and interpretation epochs must continue to satisfy their explicit identity/compatibility assumptions; bare counter equality must not be generalized into a content-integrity guarantee.

The memory claim now covers one bounded Store and transition only. Aggregate cache population, retained immutable snapshots/forks, archives and caller object lifetimes remain explicitly outside that bound. Transaction conditions include all semantic dependencies and actual atomic publication. The modern baseline roster is described prospectively, with independent admission still required. The change to increment revisions on every accepted write and to retain tombstones for absent deletes is declared as a proposed ADR extension rather than silently attributed to historical code.

The exact guaranteed behavior remains conditional on correctly supplied typed operations and the implementation/identity premises. Correct raw-language binding, successful long-horizon learning, preserved behavior under arbitrary shared-weight changes, universal retrieval coverage and matched-quality serving superiority remain unresolved. This is the appropriate conclusion to the user's guarantee question.

### Final reviewed identities

```json
{
  "reviewer": "/root/unified_guarantee_review",
  "reviewed_utc": "2026-11-23T05:17:19.073866+00:00",
  "status": "ACCEPT_CONDITIONAL_SPECIFICATION_AND_MATHEMATICAL_SCOPE",
  "architecture_sha256": "a8af3f9cf48c5302e1b1e143fd286d0c47a311de15aa03941c4fe787f82059c6",
  "mathematics_sha256": "63ea2dc485309de50663b78fe00683ae73c318fab718c100be2f4c38c163801c",
  "lean_source_sha256": "2df339e39910803713498132af1c66185900acb5d7b71f1eafb5f9709a3118ff",
  "scope": "Final mathematical/document review only; no new compiler, Python or neural execution"
}
```
