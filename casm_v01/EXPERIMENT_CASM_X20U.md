# CASM-X20U — local counterfactual credit for discrete record existence

## Status and purpose

CASM-X20U is preregistered before implementation or execution.

Frozen X20T result base:

`a122c447efc31054b61134b3271cbc282f167ddb`

X20T established a narrower boundary. Equal-weight global hard+soft answer credit improved the graph-conditioned discrete treatment from 0/6 seen-competent seeds to 4/6, with all four successful seeds also passing strong unseen `n=5,6` extension. The remaining failures were qualitatively different: one seed under-instantiated needed records, while another achieved perfect task capability but over-instantiated distractors and failed the frozen state-selection thresholds.

X20U tests one specific mechanism:

> global answer losses do not assign sufficiently local credit to individual record-existence decisions; a final-answer-only leave-one-record counterfactual objective can directly reward records whose inclusion improves the task answer and allow storage pressure to remove records whose inclusion does not help.

No live-mask, cardinality, intermediate-state, hidden-register, semantic-role, address, or causal-slice labels are used.

## Frozen base substrate

Inherit unchanged from frozen X20T/X20S/X20R/X20:

- 8 supplied candidate entities;
- version-correct temporal dependency graph;
- shared reverse-topological graph constructor;
- deterministic nonlearned candidate identity codes;
- local-equivariant gated executor;
- final-answer-only target;
- train live cardinalities `{2,3,4}`;
- unseen live cardinalities `{5,6}`;
- train depth 12;
- evaluation depths 12/24/48/96;
- batch size 128;
- 12,000 optimization steps;
- AdamW, cosine LR `2e-3 -> 2e-4`, weight decay `1e-4`, gradient clip `1.0`;
- hard evaluation threshold `g_soft >= 0.5`;
- storage coefficient `0.05`;
- no top-k, count repair, matching, rerouting, or hidden oracle;
- Python `3.11.16`;
- PyTorch CPU `2.14.0+cpu`.

Frozen X20/X20R/X20S/X20T substrate files must remain byte-identical to the X20T result base.

## Gate definitions

For constructor logit `z`:

`g_soft = sigmoid(z)`

`g_hard = 1[g_soft >= 0.5]`

`g_ST` is the exact X20R binary-forward, identity-through-soft-backward estimator.

Define:

- `A_hard = NLL(execute(g_ST), final_answer)`
- `A_soft = NLL(execute(g_soft), final_answer)`
- `S_hard = mean(g_ST)`

For each candidate record `i`, construct two counterfactual soft-execution gate vectors from the current `g_soft`:

- `g_on(i)`: identical to `g_soft` except candidate `i` is forced to `1`;
- `g_off(i)`: identical to `g_soft` except candidate `i` is forced to `0`.

Using the same executor, same batch, and same final-answer target, compute:

- `A_on(i) = NLL(execute(g_on(i)), final_answer)`
- `A_off(i) = NLL(execute(g_off(i)), final_answer)`.

The local counterfactual risk is

`L_local = mean_i [ g_soft_i * stopgrad(A_on(i)) + (1 - g_soft_i) * stopgrad(A_off(i)) ]`.

Therefore the direct derivative with respect to `g_soft_i` is proportional to `A_on(i) - A_off(i)`: if forcing record `i` on improves the final answer, local credit increases its gate; if it does not help, the storage term can remove it. `A_on/A_off` are detached only for the local gate-credit term; the executor continues to receive ordinary task gradients from the global hard/soft path.

No sign threshold, pseudo-label, temperature, live-mask target, or cardinality target is introduced.

## Frozen regimes

All learned regimes start from bit-identical compatible parameters per seed and receive identical training batches.

1. `canonical_live_mask`
   - unchanged positive control.

2. `dual_credit_replication`
   - exact best X20T treatment replication;
   - objective: `0.5*A_hard + 0.5*A_soft + 0.05*S_hard`.

3. `local_counterfactual_credit`
   - decisive graph-conditioned treatment;
   - define `A_mix = 0.5*A_hard + 0.5*A_soft`;
   - objective: `0.5*A_mix + 0.5*L_local + 0.05*S_hard`;
   - equivalently the global hard/soft task path contributes total weight `0.5` and the local counterfactual task-derived risk contributes weight `0.5`, keeping total task-loss scale approximately comparable to a single NLL-scale objective.

4. `local_counterfactual_credit_structure_blind`
   - same objective as regime 3;
   - same parameter count and observations except dependency/version message propagation is removed exactly as in prior structure-blind ablations.

The coefficient `0.5/0.5` is fixed before implementation. No coefficient sweep or adaptive retuning is allowed.

## Fresh seeds

Use six fresh paired seeds fixed before implementation:

- train `20261211`, eval `20261291`;
- train `20261212`, eval `20261292`;
- train `20261213`, eval `20261293`;
- train `20261214`, eval `20261294`;
- train `20261215`, eval `20261295`;
- train `20261216`, eval `20261296`.

No seed replacement, selective rerun, coefficient sweep, schedule adaptation, or post-hoc threshold change is permitted.

## Integrity requirements

Before scientific training, tests must establish:

1. branch ancestry includes frozen X20T result `a122c447efc31054b61134b3271cbc282f167ddb`;
2. this preregistration commit predates every X20U implementation commit;
3. frozen X20/X20R/X20S/X20T data, model, estimators, evaluators, runners, and inherited tests are byte-identical to the frozen base;
4. learned X20U regimes have identical parameter counts and bit-identical compatible initial state per seed;
5. graph-conditioned learned regimes have identical raw `g_soft` before optimization;
6. `dual_credit_replication` is numerically identical to frozen X20T `dual_credit_hard_storage` when parameters and batch match;
7. `g_on(i)` and `g_off(i)` differ from raw `g_soft` only at candidate `i`, which is exactly 1 or 0 respectively;
8. `L_local` numerically equals the preregistered expected counterfactual-risk formula;
9. `A_on(i)` / `A_off(i)` are detached for the local gate-risk coefficient while `L_local` has nonzero gradient to the constructor through `g_soft_i`;
10. the sign of the local constructor gradient matches the exact task counterfactual: when `A_on(i) < A_off(i)`, gradient descent pushes `g_soft_i` upward, and vice versa;
11. executor parameters receive task gradients from the global `A_mix` path;
12. `S_hard` retains exact X20R binary-forward value and straight-through gradient;
13. hard evaluation remains raw `g_soft>=0.5` with no repair;
14. learned regimes cannot read live mask, active cardinality, hidden trajectory, causal-slice labels, or final-answer value as constructor input;
15. no learned per-candidate gate/ID table is introduced;
16. structure-blind treatment receives identical observed candidate/program tensors except graph propagation;
17. losses, gates, gradients, counterfactual losses, and states fail closed on non-finite values;
18. inherited positive-control/executor/state-instantiation regressions remain exact through `n=2..6`, depth 96;
19. every result records exact Git SHA, Python/PyTorch versions, seeds, optimizer contract, objective identifiers, counterfactual definition, and evaluation contract.

## Frozen evaluation and classification

Evaluation is unchanged from X20T and is applied independently to every regime using raw `g_soft`.

### Positive-control validity

`canonical_live_mask` must satisfy every seed and `n=2..6`:

- hard answer-final >=99% every suite;
- hard deep step-state exactness >=95%;
- hard deep live-register accuracy >=99%.

Failure invalidates X20U.

### Seen competence

A learned regime is seen-competent on a seed only if every trained live cardinality `n=2,3,4` satisfies:

- hard and raw-soft answer-final >=99% every suite;
- hard and raw-soft deep step-state exactness >=95%;
- hard and raw-soft live-register accuracy >=99%;
- hard existence precision >=0.95;
- hard existence recall >=0.95;
- mean absolute hard record-count error <=0.25;
- mean raw soft gate on live candidates >=0.90;
- mean raw soft gate on distractors <=0.10.

Any seen IID/base hard answer-final <80% is an optimization failure.

### Strong unseen extension

A strong PASS requires **every 6/6 seed** at unseen live cardinalities `n=5,6` to satisfy the same capability and state-selection thresholds after first passing seen competence 6/6. No averaging rescues a failed seed or cell.

### Partial unseen extension

If strong fails but a seen-competent treatment has every unseen capability cell with hard/raw-soft answer >=90%, deep step/live-register >=80%, and hard existence F1 >=0.80, classify partial and report the exact boundary. Partial does not authorize reuse/merge.

## Local-credit interpretation rules

### Local counterfactual-credit repair PASS

A bounded local-credit repair claim is authorized only if:

1. `dual_credit_replication` fails the 6/6 strong criterion; and
2. `local_counterfactual_credit` passes full seen competence 6/6 and strong unseen extension 6/6; and
3. `local_counterfactual_credit_structure_blind` does not independently satisfy the same 6/6 strong criterion.

If both dual replication and local credit pass strongly, report robust state instantiation but do not claim local counterfactual credit was necessary.

If local graph and structure-blind both pass strongly, graph dependence is unresolved/shortcut-prone and reuse/merge remains blocked.

If local credit fails 6/6 seen competence, remain on the estimator/objective/credit-assignment boundary.

## Claim boundary

Even a strong X20U result establishes only learned hard-thresholded instantiate/ignore decisions over supplied candidate records from a supplied temporal program graph, using final-answer-only global and local counterfactual task credit plus explicit storage pressure.

It does not establish entity discovery, semantic role induction, identity reuse/merge, lifecycle deletion, cross-episode persistence, program/controller induction, or verifier-guided repair.

## Successor boundary

Reuse/merge is authorized only if X20U is valid and `local_counterfactual_credit` (or the replicated dual treatment) passes full seen competence 6/6 and strong unseen `n=5,6` extension 6/6 **without** the structure-blind local treatment also earning the same strong PASS.

Otherwise freeze X20U and keep the frontier on discrete state-construction estimator/objective/credit assignment.