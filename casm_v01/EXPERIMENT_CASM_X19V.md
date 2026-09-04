# CASM-X19V — soft address margin validation

## Status and purpose

CASM-X19V is preregistered before implementation or execution.

CASM-X19D established that both learned recurrent constructors remain **hard-executable and uniquely self-addressing at unseen n=5,6 on all six seeds** once the learned fixed-slot bridge is removed. The formal strong criterion nevertheless fails because fixed cosine-attention temperature `beta=16` leaves soft retrieval leakage between distinct role keys. Learned orthogonal recurrence passes strong unseen criteria on 5/6 seeds versus 1/6 for the unconstrained recurrence; the sole orthogonal failure is hard-exact but soft-inexact.

X19V is a one-shot diagnostic validation of the role-keyed memory substrate. It does not introduce another constructor architecture, allocator, learned retrieval module, learned temperature, or extra training signal.

## Question

> If the exact X19D training path is replayed at beta=16, does evaluating the resulting unchanged trained constructors with a preregistered sharper fixed beta=64 role-keyed address remove the remaining soft-execution failures at unseen n=5,6?

This is an addressing-margin validation, not a new capability-learning experiment.

## Frozen rationale for beta=64

The worst unseen orthogonal X19D seed has maximum competing role cosine similarity approximately `0.848` while self-similarity is exactly `1`. With `beta=64`, the self-vs-worst-competitor logit gap is approximately

`64 * (1 - 0.848) ~= 9.7`,

so a single worst competitor has relative softmax odds below approximately `6e-5`. This choice is fixed before execution and is intended as a near-hard but still ordinary softmax content-addressing diagnostic.

No intermediate beta sweep, tuning, per-seed beta, learned beta, or outcome-based beta selection is permitted.

## Frozen replay regimes

Replay the exact X19D model/training regimes:

1. `canonical_keyed`;
2. `frozen_random_orthogonal`;
3. `unconstrained_recursive`;
4. `orthogonal_recursive`.

Training remains exactly X19D and uses fixed `beta=16` for all soft reads/writes/decoding.

Post-training only, evaluate each finished model under two address temperatures:

- `beta16_replay` — exact X19D address contract;
- `beta64_counterfactual` — identical generated roles, executor parameters, memory values and task examples; only the scalar cosine-attention temperature changes from 16 to 64.

Hard addressing is identical between the two evaluations because hard addressing is raw argmax of cosine similarity.

## Frozen model and memory contract

Retain X19D unchanged:

- role dimension 32;
- canonical deterministic orthonormal positive control;
- frozen random orthogonal recurrent falsifier;
- learned unconstrained recurrence `normalize((I + 0.1 A) r)`;
- learned Cayley-orthogonal recurrence;
- supplied active cardinality;
- one transient record per supplied external variable;
- role-key cosine addressing;
- no physical slot bank;
- no learned role-to-slot scorer;
- no direct command-index record read/write/decode;
- no dual prices, occupancy, matching, Sinkhorn, hard masking or collision repair;
- validated local-equivariant executor;
- final register 0 only task supervision;
- no role/address labels or separation loss.

## Frozen replay budget

Use the exact X19D schedule:

- train n in `{2,3,4}` only;
- schedule `2,3,4` repeated;
- depth 8;
- batch size 128;
- 10,000 AdamW steps;
- weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`;
- global gradient clipping `1.0`;
- eval n `2..6`;
- depths `8,12,24,48,96`;
- eval_n 256.

Use the **same six replay seeds** as X19D because the primary requirement is exact replay of the already-frozen training path, not a new robustness panel:

- `20261161 / 20261241`;
- `20261162 / 20261242`;
- `20261163 / 20261243`;
- `20261164 / 20261244`;
- `20261165 / 20261245`;
- `20261166 / 20261246`.

No seed replacement or selective rerun is permitted.

## Replay integrity requirement

Before interpreting beta64, the beta16 replay must reproduce the frozen X19D outcome class on every seed:

- canonical positive control remains exact;
- both learned recurrences remain seen-competent on all six seeds;
- unseen hard execution and hard self-addressing remain exact on all six seeds for both learned recurrences;
- orthogonal beta16 strong unseen status remains 5/6 and unconstrained beta16 strong unseen status remains 1/6;
- frozen-random beta16 strong unseen status remains 4/6.

Exact floating-point equality to archived X19D JSON is not required across runner environments, but any change in those discrete pass/fail classifications invalidates the beta64 interpretation.

## Implementation integrity requirements

Before training, tests must establish:

1. X19D model parameterization and beta16 training path are unchanged;
2. beta64 is never used in a train-time loss or forward pass;
3. no role `r4+` or n=5/6 learned execution occurs before step 10,000;
4. beta64 evaluation changes only the softmax scale over the same cosine logits;
5. generated roles and model parameters are bit-identical between beta16 and beta64 post-training evaluation views;
6. hard-address argmax is identical for beta16 and beta64;
7. record-order invariance and no-direct-index addressing remain intact;
8. frozen-random constructor tensors remain frozen;
9. orthogonal recurrence remains Cayley-orthogonal within `1e-5`;
10. all outputs remain finite.

## Frozen classification

### X19D replay prerequisite

If the beta16 replay discrete classifications differ from frozen X19D as specified above, classify X19V **INVALID REPLAY** and make no beta64 claim.

### Beta64 strong soft-address validation

For a regime to pass beta64 validation, every seed at unseen n=5,6 must satisfy the same X19D strong criteria:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- every active role uniquely hard-self-addresses;
- mean soft self-address probability >=0.90.

No averaging rescues a failed seed/cell.

### Interpretation

If `orthogonal_recursive` passes 6/6 at beta64 while beta16 replay remains 5/6, conclude that X19D's remaining formal orthogonal failure was a fixed soft-address-margin artifact under this diagnostic memory substrate.

If `frozen_random_orthogonal` also passes 6/6 at beta64, conclude more strongly that this controlled benchmark does **not require learned role semantics** once supplied variables receive transient role-keyed records; sufficiently separated extensible identity codes are enough. The next scientific frontier should then move to learned **state instantiation/reuse/deletion decisions**, not better role-code generation.

If orthogonal passes 6/6 but frozen-random does not, conclude that learned noncontractive role geometry contributes useful addressing margin even though semantic-role learning is still not established.

If orthogonal fails beta64 on any seed despite valid replay, do not advance to X20; constructor/memory-key geometry remains insufficient.

## Claim boundary

Even a 6/6 beta64 result does not establish variable discovery, cardinality discovery, semantic role induction, learned memory creation, program induction, persistence, or verifier-guided repair. It only validates that recurrently generated or frozen noncontractive role identities can support robust executable role-keyed state through unseen role positions under a sharper fixed content-addressing substrate.