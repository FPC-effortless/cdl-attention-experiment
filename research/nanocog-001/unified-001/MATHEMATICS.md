# Mathematical study of COG-UNIFIED-001

## Result

The fused architecture supports conditional structural guarantees. It does not guarantee all desired learned capabilities, and some unrestricted combinations of requirements are mathematically impossible. This conclusion is a result of the study, not an unfinished proof that should be silently treated as true.

The verification levels are deliberately different:

1. **Lean-checked abstract mathematics:** 46 theorem statements, including structural laws, conditional composition and explicit counterexamples. Lean 4.19.0 checked the source without `sorry`, `admit`, or custom axioms. A separate agent process independently compiled the same source and generated its own per-theorem axiom audit.
2. **Written mathematics:** general finite-information bounds, conditional probability bounds, full transaction conditions, floating-point/stability caveats and cost analysis. These are not all formalized in Lean.
3. **Executable reference checks:** constructed tests of a Python exact-state kernel. Passing these is not a Lean refinement proof and not model performance.
4. **Learning and deployment:** no new neural fit, protected evaluation, language benchmark, accelerator timing or model promotion ran. These guarantees remain open.

The complete independent mathematical audit is `review/guarantee_analysis.md`. The exact theorem list and compiler receipts are in `proofs/`.

## 1. Formal objects and event semantics

Let K be logical keys, V typed values and W a store. A key includes namespace, entity and relation. Abstract cells are values, references and absence/tombstones. The reference additionally carries revision, schema, evidence and producer metadata.

An accepted event acts by a finite update. For SET:

\[
\operatorname{SET}(W,k,v)(j)=
\begin{cases}v,&j=k,\\W(j),&j\ne k.\end{cases}
\]

This gives read-after-write, overwrite and frame laws:

\[
\operatorname{SET}(W,k,v)(k)=v,
\]
\[
\operatorname{SET}(\operatorname{SET}(W,k,v_1),k,v_2)
=\operatorname{SET}(W,k,v_2),
\]
\[
j\ne k\Rightarrow\operatorname{SET}(W,k,v)(j)=W(j).
\]

These statements are machine checked as `write_read_same`, `write_overwrite` and `write_read_other`. They prove exact execution of an admitted event. They do not prove that a learned parser chose the right event.

DELETE replaces the current record with an explicit absent/tombstone representation. A current-state read cannot silently fall back to an older value. Snapshot COPY first resolves the source and then stores a value; live LINK stores an address. Lean checks both behaviors and COPY refusal on unresolved sources.

### Why COPY and LINK cannot share a vague preservation promise

Suppose A links to B and B holds red. Updating B to blue leaves A's stored cell unchanged, but reading A returns blue. This is a desired live-reference effect, not an executor corruption. Conversely, a snapshot copied from B remains red until that snapshot itself is changed.

The corresponding Lean counterexample is `live_link_answer_changed`, paired with `link_cell_preserved`. The design must distinguish byte/cell preservation, dependency-aware answer preservation, and preservation of learned behavior.

## 2. Exact retrieval is conditional on grounding and decoding

Let the model infer a key \(\hat k\), let the intended key be \(k^*\), and let d be a decoder. The elementary composition law is

\[
\hat k=k^*,\quad W(k^*)=v^*,\quad d(v^*)=y^*
\quad\Longrightarrow\quad d(W(\hat k))=y^*.
\]

`grounded_read_is_correct` checks this implication. The premises are visible. A theorem that assumes \(\hat k=k^*\) is not a proof that a trained binder finds \(k^*\). A theorem that assumes \(d(v^*)=y^*\) is not a proof that a free-form neural decoder preserves a fact.

For a restricted typed answer format, a deterministic renderer can discharge the value-to-text step. For arbitrary natural language, factual faithfulness remains a learned or externally verified property. A flawless store can hold an incorrectly inferred value, and an impeccable provenance chain can identify an incorrect source.

## 3. Dependency-aware preservation and caching

For a deterministic read q on snapshot W, let \(D_W(q)\) include every cell inspected, including absent keys. If another snapshot W' agrees on every such cell, the bounded LINK interpreter produces the same result. Lean proves this by induction over the fuel-bounded read in `unchanged_read_path_preserves_resolution` and derives `bounded_link_reader_locally_depends_on_path`.

The theorem is local to a snapshot and a specific interpreter. A global assertion that one fixed set of keys suffices for every possible state would be unnecessarily strong for dynamic reference chains.

For a generic cached computation F, sound reuse needs

\[
\operatorname{Identity}(q,\theta,\text{tokenizer},\text{policy},
\text{schema},\text{stream},\text{all dependencies})
\]

to match. If F includes learned candidate selection, the candidate/index state is a dependency. If F includes a neural output, recurrent state h, broker packets P and relevant randomness are dependencies. A query and stream name alone are insufficient. The implemented cache is deliberately limited to pure typed resolution.

### A discovered fork counterexample and the resulting repair

From `(A=red, revision=1)`, two legal immutable branches may produce `(A=green, revision=2)` and `(A=blue, revision=2)`. Equal numeric revisions do not imply equal contents. A cache or COPY transaction guarded only by the number 2 can use the wrong branch's value.

The reference now guards each dependency by its **full immutable cell observation**, including revision, payload, schema, evidence and producer, or exact absence, together with epochs. This avoids making an unproved hash or globally unique ID assumption in the semantic implementation. A production compact guard must establish incarnation/content integrity across fork and restore; equality of a bare counter is not enough. The work to compare or authenticate those observations must be charged.

The Lean version-based cache theorem has an explicit `versionSound` premise. The fork example explains why that premise cannot be omitted. The reference's stronger equality is not by itself a machine-checked refinement of that theorem.

## 4. Transactions and safe parallelism

A transaction's proposal is evaluated on one immutable snapshot. Commit must validate its entire read set, write set and relevant authority/policy/schema epochs against the current snapshot, then publish all writes or none. Snapshot comparison and publication must be atomic in an actual shared implementation. The pure Python transition models this operation but does not implement distributed coordination or durable storage.

For a structural invariant P, if every admitted candidate satisfies P and rejects preserve the old state, induction preserves P. This relies on a sound checker and atomic publication. The Lean Boolean commit laws only establish branch choice; they are not a proof of the complete transactional implementation.

For transactions f and g with complete read sets \(R_f,R_g\) and write sets \(U_f,U_g\), sufficient commutation conditions are

\[
U_f\cap(U_g\cup R_g)=\varnothing,
\qquad
U_g\cap(U_f\cup R_f)=\varnothing.
\]

Then neither transaction changes the other's inputs or destinations. On each written key the final value is its transaction's unchanged proposal; elsewhere both preserve the initial store. This proves extensional semantic-store equality. Audit order, timestamps, global counters and unique commit IDs may differ.

Disjoint destinations alone are insufficient. If one action writes A from B and another writes B from A, order can change the result. `disjoint_targets_still_interfere` is a checked counterexample. The reference also checks a write-skew pattern and rejects the second transaction after its observed input changes.

Live references, absent-key predicates, shared capacities, counters, authority and index maintenance all enlarge read/write footprints. Their dependencies cannot be omitted to manufacture parallelism.

## 5. Causality and conditional whole-run correctness

Let a deterministic step depend on current state and current token only. Repeated application gives

\[
S_{t+1}=T_\theta(S_t,x_t).
\]

With the same initial state, fixed parameters and compatible random stream, equal input prefixes produce equal prefix states and outputs. Lean checks fold composition and suffix independence. The candidate preserves read-before-write: its current output uses the old record store, then the current packet updates the next store.

A stronger compositional theorem uses a relation R between actual and ideal states. If

\[
R(S_0,S_0^*)
\]

and, for every relevant input,

\[
R(S,S^*)\Rightarrow
R(T(S,x),T^*(S^*,x)),
\]

then every finite run preserves R. If related states have matching readouts, their final outputs match. This induction is checked by `run_preserves_simulation` and `simulated_run_has_same_readout`.

This is the correct shape of a whole-system guarantee. Its critical step-simulation premise has not been proved for the Python kernel, tokenizer, neural binder, decoder, training procedure or hardware kernels. The proof identifies the obligation; it does not remove it.

## 6. Parallel local recurrence and numerical stability

For an affine local branch

\[
h_t=A_t h_{t-1}+b_t,
\]

define transform composition

\[
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,A_2b_1+b_2).
\]

Associativity follows from multiplication associativity and distributivity. The checked Lean version uses exact scalar natural arithmetic; the written matrix/diagonal derivation follows the same algebra. With diagonal \(A_t\), composing transforms is linear in width, avoiding the generic matrix-matrix overhead.

The useful scan requires coefficients to be available independently of the same layer's evolving state. If a gate uses a newly retrieved value or a same-layer hidden state, the bounded affine representation generally no longer applies. Saying all functions compose associatively does not produce a practical scan algorithm. Record commits and conflicting actions still require causal ordering.

Floating-point arithmetic is not exactly associative. A kernel must separately validate numerical error and the effect on discrete route decisions. A small numeric change can flip argmax near a tie. No bitwise or speed guarantee follows from the real/integer algebra.

For exogenous coefficients with \(\lVert A_t\rVert\le\rho<1\), equal later inputs imply

\[
\lVert h_{t+k}-\tilde h_{t+k}\rVert
\le\rho^k\lVert h_t-\tilde h_t\rVert.
\]

That is a stability bound and also an attenuation bound for old differences. It cannot simultaneously be advertised as unrestricted exact long-term memory. If gates depend on h, the full Jacobian includes gate derivatives; a sigmoid range alone does not prove contraction.

## 7. No-forgetting requires a declared scope

An exact protected-cell policy can prevent writes to protected records. Output preservation additionally requires preserving all live dependencies and the interpretation/routing functions.

For a frozen base function and a residual update,

\[
f_{\mathrm{new}}(x)=f_{\mathrm{old}}(x)+r(x),
\]

the sufficient preservation condition over a protected domain D is

\[
\forall x\in D,\ r(x)=0.
\]

The formal scalar development checks this condition and an isolated-route variant. For a fixed linear head with protected features H, the corresponding algebraic condition is \(\Delta W H=0\). It protects that feature span only while upstream features and all relevant routes remain unchanged; this latter vector statement is written mathematics, not part of the Lean module.

A finite regression suite does not prove the universal premise. Two functions may agree on every tested input and differ on one untested input. The Lean development constructs exactly this counterexample. Low parameter movement, low rank, an edit confined to a few neurons or a passed regression suite do not automatically prevent forgetting.

Learning new facts in isolated exact records can preserve other records. Learning arbitrary new behaviors by changing shared parameters creates a separate non-interference problem. That distinction remains explicit in the fused design.

## 8. Finite information imposes a hard limit

Assume the total retained state contains at most B bits, with no external retained record, and must exactly recall any of n arbitrary binary facts. There are \(2^n\) assignments but at most \(2^B\) states. If n>B, two distinct assignments share a state. They differ at some queried key. The decoder receives the same state and query but requires different answers; it cannot be correct on both.

For q possible values per independent fact, the exact information requirement is at least

\[
B\ge n\log_2 q
\]

bits, before key, metadata or retrieval overhead. This counting argument is a written proof. Lean checks a concrete collision and the general implication from an encoding collision to failure on distinguishable answers, not the entire cardinality derivation.

An external archive avoids the fixed-total-state premise by retaining more bits. Its size and access work must be counted. The coherent design is bounded fast state plus an accounted growing archive, or a bounded retention promise with explicit refusal/eviction. Unlimited exact recall, fixed total bytes and unrestricted input information are incompatible.

## 9. Sparse routing cannot guarantee unrestricted recall by budget alone

In an unstructured collection of N candidates, suppose relevance can only be learned by inspecting a candidate and no appropriate complete index or sufficient statistic is already available. A method inspecting fewer than N items can miss a relevant item: two collections can agree on every inspected item and differ only at an uninspected location.

An exact index for the correct key changes those premises. It can support exact lookup when the key is available and the index is complete. Learned semantic retrieval does not become complete merely because it returns K candidates. `concrete_sparse_miss` and `omitted_candidate_is_unread` formalize a finite obstruction.

Budget caps guarantee bounded declared work, not universal coverage. The architecture must expose an unresolved result or a separately charged escalation. Always refusing avoids wrong answers but fails usefulness; coverage, correctness and cost must be assessed jointly.

## 10. Binding and paraphrase invariance remain learning obligations

Separate entity/relation/value heads, support labels and counterfactual losses can improve a model. They do not prove that a different phrase maps to the same intended event.

For a specified set of meaning-preserving transformations G, a true invariance claim is

\[
F(gx)=F(x)\quad\text{for all }g\in G.
\]

Entity renaming usually calls for equivariance instead: renaming inputs and keys must consistently rename outputs. Such laws can be enforced or proved for restricted transformations. Training on some examples of them does not establish the universal statement, and arbitrary natural-language paraphrases are not a supplied formal group with a trusted membership oracle.

Ambiguous observations create an additional limit: identical observed text can be compatible with worlds requiring different actions. No architecture can always infer the hidden world correctly without more evidence, an assumption narrowing the worlds, or abstention.

The 001A support-supervision and 001D readout-repair gains motivate components of this proposal. Their original readers and data contracts differ from the strict fused candidate. Their percentages are not inherited guarantees.

## 11. Conditional reliability is not a numerical guarantee without calibrated bounds

Let E_i be failures of acquisition, binding, retrieval, decoding, admission, implementation or external verification. If the final answer is correct whenever none occur, then

\[
\Pr(\text{wrong answer})\le\Pr\left(\bigcup_i E_i\right)
\le\sum_i\Pr(E_i).
\]

No independence assumption is needed. But every bound must apply to the actual operating distribution, including adaptive routing and selective acceptance. A development-set error rate is not automatically such a bound under shift.

If each stage has a conditional success lower bound \(1-\epsilon_i\) given all earlier successes, the chain rule gives a product lower bound. Multiplying unrelated marginal accuracies from separate experiments would be invalid. Likewise, two agreeing verifiers are not independent merely because they have different names or processes.

No empirical epsilon values for this fused architecture have been established, so this study supplies the form of a guarantee, not an invented end-to-end percentage.

## 12. Resource and economic analysis

For n input tokens, Q queries and U events, a complete accounting must include

\[
C = C_{\mathrm{load}} + nC_{\mathrm{local}}
 + U(C_{\mathrm{bind}}+C_{\mathrm{validate}}+C_{\mathrm{write}}+C_{\mathrm{index}})
 + Q(C_{\mathrm{candidate}}+C_{\mathrm{read}}+C_{\mathrm{decode}})
 + C_{\mathrm{reset/fork/recovery}}+C_{\mathrm{fallback}}.
\]

The coefficients vary with model width, vocabulary, hardware, batching, store size and output length. A dense vocabulary head costs order Vd per scored token; learned dense projections may cost order d squared. Removing history attention does not remove these terms. With an external archive of R records, index/build/update work must be explicit.

The reference uses linear tuple lookup. A link read can therefore cost order HM, even though it follows at most H links. It is not an optimized indexed implementation. Full immutable dependency observations also require comparison work and retained metadata. A smaller number of logical operations is not automatically lower wall time.

A bounded fast working set can have a size bound in terms of M, P, H, cache/procedure sizes, the number of live snapshots/forks, payload widths and metadata caps. The reference bounds one Store instance and individual transitions, not all objects retained by a caller or aggregate process memory. Total retained archive information still grows. Fixed-width counters must reject exhaustion or use a specified rollover protocol; the reference rejects a 64-bit revision overflow rather than wrapping.

For reuse, if build cost is B, recomputation costs c and validation costs v, a simple sufficient economic condition after N valid uses is

\[
N(c-v)>B,\quad c>v.
\]

Real workloads must add cache misses, invalidation, rebuilding and memory opportunity cost. Even a perfectly correct cache can be economically harmful on uniform or changing workloads, as the prior research already showed.

The existing comparison protocol requires quality qualification before efficiency claims. For a declared deployment volume, report amortized training, complete prefill/generation, maintenance and recovery in compatible units. The user's 12:1 inference preference is a weighting choice, not a proof that the candidate is faster or permission to trade away required quality. No speed ratio from the separate historical components can be multiplied to predict a fused-model ratio.

## 13. Guarantee ledger

| Requirement | Result of this study | Remaining premise or gate |
|---|---|---|
| Exact typed SET/DELETE/COPY/LINK | Abstract laws checked in Lean; reference tested. | Correct admitted operation, valid types, executable refinement. |
| Preserve untouched raw records | Checked frame law. | Protected key set and storage semantics correctly implemented. |
| Preserve unrelated answers | Conditional dependency theorem; bounded LINK path theorem checked. | Complete dependencies and fixed interpretation, including h/P for neural outputs. |
| Safe cache reuse | Conditional laws checked; fork-aware reference repaired. | Exact/full identities or sound compact identities, compatible epochs, pure typed scope. |
| Prefix causality | Causal-fold theorem checked. | Actual tokenizer, packer, routes and kernels obey prefix-only interface. |
| Finite-run ideal behavior | Simulation theorem checked. | One-step and readout simulation obligations remain unproved for learned implementation. |
| Atomic valid state changes | Written conditional invariant argument; reference falsifiers. | Full read validation and atomic publication; no deployed storage proof. |
| Parallel exact local algebra | Exact affine law checked; semantic write conditions analyzed. | Input-dependent coefficients, nonconflicting dependencies, numeric implementation parity. |
| Bounded declared read/write work | Bounded LINK law checked; reference caps/refusals. | Discovery, validation, payload size and all fallback work must also be bounded. |
| Exact unbounded recall with fixed total bits | Impossible for unrestricted independent facts. | Allow growing storage or narrow retention. |
| Complete arbitrary semantic retrieval at fixed candidate count | Not guaranteed; explicit obstruction. | Complete usable index, a narrower query class, or measured coverage/escalation. |
| Correct binding on every unseen phrase | Not proved or established. | Restricted semantic contracts or new empirical generalization evidence. |
| Universal no-forgetting under arbitrary weight changes | Not guaranteed; finite-suite counterexample checked. | Structural isolation or a domain-wide preservation proof. |
| Guaranteed training convergence/useful memory | Not proved. | Learning signal, representability, optimization and adequate data. |
| Better LLM quality and serving cost | Not established for this fusion. | Integrated trained model and fair quality-constrained comparisons. |

## 14. Decision

Accept this as a coherent architecture specification, abstract formal development and falsified/repaired semantic reference. Do not call it a guaranteed general-purpose attention replacement. The next substantial obligation is a native token-to-event-to-answer implementation with jointly trained interfaces and adequate matched controls; the next theorem obligation is refinement of the executable semantics, not a larger count of unrelated structural lemmas.
