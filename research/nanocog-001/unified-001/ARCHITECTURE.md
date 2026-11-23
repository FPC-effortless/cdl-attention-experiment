# COG-UNIFIED-001: compositional state with learned binding

Status: prospective architecture and executable semantic reference. No trained unified model, language-model result, serving benchmark, or architecture promotion is claimed.

This is a versioned composition proposal within the existing COG-PULSE / CSM / nanoCOG research program, not a replacement history or a claim that historical checkpoints share a representation. The source baseline is the attention-comparison proposal at `1a91168b216a49543cc839153fa299a262320730`; its inspected nanoCOG ancestor is `8374f72ba9324fa4a11d0d3ecb40a6db0e311c73`. PR88 subsequently advanced to `39be434ff160d4e6507df8d2114979ba5529caff`. Existing results, protected cohorts and qualification decisions remain unchanged.

## 1. Design decision and intended outcome

Combine the mechanisms at a common interface: causal local computation, factorized binding, supervised selection and writing, stable typed records, bounded retrieval, aligned value decoding, dependency-aware reuse, and selective proposed updates. Train a shared representation and these interfaces together on one declared curriculum. Do not concatenate historical weights or multiply their separate reported gains.

The desired system must receive raw tokens, infer its own entities/relations/updates, answer both sides of a counterfactual, preserve unrelated facts, learn corrections, and provide complete quality and cost accounting. Exact typed inputs are allowed only in the semantic reference and oracle diagnostics, never silently in the native learned arm.

The primary candidate preserves nanoCOG's no-token-attention boundary: no token-to-token QK attention, dense historical KV attention, or borrowed SSM implementation/checkpoint. It uses its own causal recurrence, hard packet routes and predicted categorical record addresses. General learned hard selection is still a form of selective retrieval; avoiding softmax attention does not make retrieval intrinsically cheaper or correct. A sparse dot-product record reader may be tested as an explicitly named hybrid comparator, not inserted into the primary candidate while retaining the same claim.

## 2. State and common representation

At input step t, before committing that token's proposal, the state is

    S_t = (h_t^in, P_t, W_t, E_t, I_t, C_t, e_t).

- h: fixed-width causal local recurrent state and the minimal previous-token state needed by the common backbone.
- P: bounded publish/subscribe packets. Each packet has an owner stream, topic, producer revision, causal step and content. Packet routing is not semantic identity.
- W: active exact typed records, addressed by a full logical key (namespace, entity, relation). Record identity is separate from mutable payload and physical row position.
- E: append-only evidence/revision archive, when enabled. Its size is charged and may grow; it is not hidden fixed-size memory.
- I: derived indexes mapping complete keys or declared searchable features to records/evidence. Hashes accelerate lookup, but full-key equality is verified; collisions cannot silently merge records.
- C: optional cached answers, routes and compiled procedures with dependency certificates.
- e: stream/session, authority/policy, tokenizer, encoder/decoder and storage-schema revisions. These are compatibility checks, not evidence of semantic truth.

A record is (full key, payload kind, payload, per-key revision, evidence references, producer revision). Payload kinds are VALUE(v), LINK(k), and TOMBSTONE. Values used by the exact interpreter have a fixed schema and stable meaning. Raw evidence and producer identity are retained for re-encoding. An arbitrary old latent vector is not assumed to retain its meaning after the neural encoder or decoder changes.

Use disjoint channels for entity, relation, value and temporal/operator information. These are architectural interfaces and supervised factors, not claims that the network discovers human semantic roles. A learned alias resolver must establish which mentions refer to the same entity. Stable storage keys do not prove that the resolver assigned the correct key.

## 3. One causal processing loop

1. Encode the current token with the same tokenizer and compatible embedding, output head, residual, normalization and MLP interfaces as the retained baseline. No future token, gold operation or evaluator field enters this path.
2. Compute the local recurrent feature. A scan-compatible branch has

       h_t = a_theta(x_t) * h_(t-1) + b_theta(x_t),

   with elementwise multiplication and coefficients independent of the same layer's evolving state. Inputs from a lower causal layer are allowed. A gated nonlinear recurrence is a separately labeled variant and does not inherit the affine scan theorem.
3. Read old broker packets through a bounded learned hard route. Keep a direct current-event feature path available to binder/write heads; do not force current-event meaning through a stale broker selection. The earlier bypass improved writes but not answers, so this is an explicit information path and ablation, not a claimed repair.
4. Infer an event boundary, entity/relation identity, operation, value and evidence span from causal features. Training supplies available support/write labels; inference does not. Uncertain or malformed proposals may be rejected or returned as unresolved. Rejection reduces coverage and must be scored.
5. Infer a query key or bounded candidate plan. An exact complete key can use exact indexing. An incomplete semantic query uses a bounded candidate generator and learned selector; if sufficient evidence is not reached under the declared budget, return unresolved or use a separately charged fallback. A fixed K does not guarantee retrieval recall.
6. Read the pre-commit W_t, following at most H live links, and retain the transitive key/version dependencies actually inspected. Cycles, missing targets, incompatible payload versions and exhausted hop budgets yield explicit statuses.
7. Produce next-token logits using the current local feature, old broker reads and compatible retrieved content. A selected exact value may instead be rendered by a deterministic typed renderer for supported structured outputs. An unconstrained language decoder receives no theorem that it will preserve a retrieved fact.
8. Validate and atomically apply an accepted proposal to produce W_(t+1). Publish the current packet after reading old packets. The existing read-before-write contract is preserved. A query-only step issues HOLD. Event batches are a prospective extension, with an explicit read-set/write-set contract.
9. Invalidate dependency certificates as necessary. Log the observed commit and its outcome. A later learning update is a proposal checked against the relevant retention and capability criteria, not an automatic privilege of the model.

The output at t can depend on x_t and the prior prefix but cannot depend on a future input. Reading before the current commit is intentional. The local feature still contains the current token. A same-event read-after-write variant would be a distinct causal contract and must not be compared as if it had identical information access.

## 4. Typed operations

| Operation | Exact semantics | Important boundary |
|---|---|---|
| HOLD | Preserve the store. | No update is inferred from absence of a proposal. |
| SET(k,v) | Replace k with VALUE(v), increment k's revision. | Correct execution does not establish that v was inferred correctly. |
| COPY(dst,src) | Resolve src on the declared snapshot and store the resulting VALUE at dst. | Subsequent changes to src do not alter the copied value. Failed resolution rejects the copy. |
| LINK(dst,src) | Store a live reference to src. | Subsequent source changes may intentionally change answers through dst. |
| DELETE(k) | Replace k with TOMBSTONE, preserving a monotone revision history. | Deletion does not erase archived evidence unless a separately specified erasure operation is applied. |
| QUERY(k) | Resolve a value on the pre-write snapshot with a hop cap. | Missing, cyclic and over-budget states are distinguishable from a valid zero/empty value. |

Every transaction carries observations of every read and written key, including absent keys. For a read-through-link operation, this includes all dereferenced keys. An absent key begins at revision zero; deletion never resets a prior key to zero. Counters alone are insufficient across forks: two branches can assign revision two to different values. The reference therefore compares the full immutable cell (revision, payload, schema, evidence and producer), or exact absence, as well as epochs. A production compact identity must bind incarnation/branch and content with an explicit integrity assumption; a bare counter is prohibited. Comparing these observations has a cost.

A commit is allowed only if the authority, policy and schema/interpretation epochs and all declared observations still match, or an explicitly proved compatible transition revalidates them. All writes then apply together on one immutable snapshot. The reference implements a single-process atomic transition, not a distributed database, durable write barrier, crash-consistent storage engine, or proof that a caller disclosed its entire semantic read set. An actual shared store must publish the compared-and-updated snapshot atomically; two independent callers cannot merely overwrite a common pointer after comparing obsolete snapshots.

Concurrent transaction claims require independent read/write footprints. Disjoint writes alone are insufficient: one transaction may read what the other changes. The parallel theorem is about final extensional store contents under those assumptions. It does not promise identical log order, global timestamps or floating-point bit patterns.

## 5. Bounded fast memory and growing exact evidence

Let M be the active record capacity, P the packet capacity, K the retrieved candidate cap, H the link-hop cap, C the cache-entry cap and F the number of live snapshots/forks. The fast state is bounded only if those capacities, procedure sizes, payload widths and metadata limits are bounded. A conceptually fixed-width hidden vector represented by infinite-precision real numbers is not a finite-bit implementation. The reference bounds one Store and individual read/transaction work; callers may retain many immutable Store/cache objects. It does not enforce an aggregate process, cache-population or live-session memory limit. Such ownership/lifetime budgets are required for a serving memory guarantee.

Exact lifetime storage is an optional growing archive. If the archive holds R records, its memory and index cost are O(R) in stored information, apart from indexing overhead. Reads and writes must charge serialization, index maintenance, transfers and any retrieval escalation. A full exact key can admit O(log R) comparison-based lookup, or expected hash-table bounds under explicit assumptions; arbitrary semantic queries do not inherit those bounds.

Eviction must be visible and policy-controlled. A protected active record cannot be silently overwritten to make space. If capacity is exhausted, reject, spill to an accounted archive, or explicitly evict an eligible record and invalidate dependent caches. Report success and refusal separately. The architecture deliberately does not promise unlimited exact facts, fixed total memory, and exact unrestricted answers simultaneously.

## 6. Reuse and procedural skill

Use demand-aware caching from the successful reuse work. The executable reference caches only pure typed resolutions. Its cache key binds the exact query, hop cap, stream, model/tokenizer/schema/policy revisions, and every relevant immutable cell observation. Equal numeric revisions from different forks are not enough. If an answer depends on absence, exact absence is part of its certificate. If selection depends on the whole candidate index, the index generation is also a dependency. A local top-one cache cannot claim safety from only the selected record's version when a newly inserted competitor could change selection.

A neural-output cache additionally depends on current local state h, broker packets P, all other causal input features and random-state semantics. It must bind their exact sufficient identity or have a separate proof that the cached output is independent of them. A stream identifier alone is insufficient. Such a neural-output cache is not implemented or qualified here; the cost of establishing identity is part of any later reuse comparison.

Compile a procedure only after its measured construction cost can plausibly be recovered over repeated queries. Recheck the procedure's preconditions and dependencies at use time. A learned ordering heuristic can reduce work in a supplied search algorithm; neither A* nor the Sudoku/chess machinery is a compulsory language-model layer. The transferable mechanism is cost-aware ordering and validated reuse.

If build cost is B and each valid reuse saves c minus v, where v is validation cost, reuse is favorable after N uses only when N(c-v) > B, with c>v. Add invalidations, rebuilds, memory opportunity cost and misses to the actual deployment equation. The 1.94x puzzle or 3.29x hot-goal figures are historical workload results, not predictions for this architecture.

## 7. Training and selective learning

Train one shared native system with the same allowed observations as its comparators. A proposed loss family is

    L = L_LM + lambda_s L_support + lambda_w L_write
        + lambda_b L_binding + lambda_p L_preserve
        + lambda_v L_value + lambda_c L_cost.

These coefficients are protocol choices to freeze using TRAIN/CAL only. No theorem guarantees that minimizing this nonconvex loss yields semantic binding or a speed advantage.

- L_support: supervised selection logits, carrying the 001A insight. For the strict candidate this trains categorical key/route decisions; it does not import the dot-product clause reader or inherit its measured gain.
- L_write: boundary, operation, key, commit and payload supervision, with labels and exposures separately counted.
- L_binding: changed-target counterfactual pairs, unseen-key combinations and role reversals, with both histories required correct.
- L_preserve: unchanged third entity, no-op, old supported skills and natural-language rehearsal. A finite preservation panel is evidence only on that panel.
- L_value: align stored value representations with the actual output interface, including matched paraphrase training. Keep a correct-source diagnostic to separate selection and value decoding.
- L_cost: a measured or justified differentiable surrogate for work. A small surrogate loss is not a latency result.

Hard discrete writes do not obtain ordinary gradients through argmax. Auxiliary supervised logits, a declared straight-through estimator, policy-gradient method or another explicit estimator is needed. Every claimed learning path requires gradient/estimator validation. The existing packet route's dense training surrogate must be counted in training cost even if inference is sparse.

Long-delay acquisition needs retained information and a valid training signal for the relevant writes and reads. Long backpropagation is one route; local write supervision or another validated credit mechanism may supply that signal without differentiating through the entire delay. The earlier longer-horizon and high-amplitude bank experiments did not solve binding. Do not use activation magnitude as the objective's success criterion. Include sufficiently diverse independent relational events, controlled compositions and natural-language rehearsal before interpreting a failure as an architectural limit.

Selective weight adaptation is an optional, separately gated subsystem. Freezing records protects their bytes, not the behavior of a changed decoder. Exact preservation over a declared input set requires structural isolation or a proven zero residual there; a passed finite regression set cannot guarantee all future inputs. Model updates invalidate dependent caches and old latent interpretations until compatibility is established.

## 8. What is fused, and what remains a comparator

The evidence map provides immutable sources and limits for each prior result. The primary architecture combines the successful interfaces or training principles; it does not claim every historical implementation belongs in its critical path.

- Keep recurrence, stable state, support supervision, factorized bindings, value-reader alignment, direct causal event access and demand-aware reuse as explicit components or matched switches.
- Preserve the successor lookup as a bounded, document-scoped optional exact retrieval arm; its literal-copy benefit must be separated from learned semantic memory.
- Keep the original PULSE controls, local-only retrained control, exact typed ceiling and text-facing simple controls.
- Keep dot-product record attention as a separately labeled hybrid; keep modern attention and recurrent challengers under the existing comparison protocol.
- Keep specialized planning solvers outside the mandatory token mixer. Their learned scheduling/reuse interfaces may be called when a task supplies a legitimate planning environment.
- Keep Veritas as the external experimentation, scoring and verification system. It is not a neural layer and cannot confer correctness on a false inferred fact merely by logging it.

## 9. Qualification needed after this study

First check semantic contracts and source refinement; then integrate the pinned tokenizer/backbone/trainer/serving interfaces; then demonstrate TRAIN/CAL adequacy under frozen criteria and freeze independent cohorts before fitting. Compare the unified candidate against local-only, removal/replacement controls, original PULSE where recoverable, simple exact/lexical references, and the proposed modern-attention roster after each comparator is independently admitted.

Use the inherited three tracks: common-data mechanism effects, training-budget frontier, and quality-constrained complete serving cost. Report native and oracle results separately. Retain failed seeds, both roles, all counterfactual orientations, untouched-third accuracy, full state, coverage, absent queries and stale answers. Treat the already exposed BIND/PULSE cohorts as diagnostics, not fresh qualification.

No scale, model size, training length or dataset is adequate merely because it was chosen. Admission must justify that it can discriminate the claimed mechanism, show competent controls and adequate training, and record actual tokenizer, parameter, data, exposure, resource and uncertainty accounting. Prefer inference quality/cost with the user's 12:1 priority only after quality qualification and with compatible units. New runs are not executed by this proposal.

## 10. ADR: extensions and compatibility

ADR COG-UNIFIED-001 is proposed, not a replacement of accepted upstream behavior. It extends the diagnostic single-packet/one-link store to explicit bounded multi-hop resolution and read-set-validated multi-record transitions in a separate namespace. It preserves causal pre-write reads and stable identities. It adds producer-compatibility and dependency-closure obligations. Unlike the old material-change-only counter, the new reference increments a revision on every accepted write, including an identical SET or repeated DELETE; deleting an absent key creates a tombstone and consumes a slot. This conservative revision/capacity rule avoids silent identity reuse and requires separate compatibility review. It does not change old checkpoints, results, success criteria, exposed data or the strict candidate's attention boundary.

The runnable reference exercises typed semantic operations and fusion boundaries with constructed inputs. Neural tokenization, entity discovery, learned binding, real-text output and efficient accelerator kernels remain implementation/learning obligations. The formal development proves an abstract subset; a separate refinement proof would be needed to certify the Python implementation, and a further empirical program to establish learned behavior.
