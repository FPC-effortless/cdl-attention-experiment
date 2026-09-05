# CASM-X20T — counterfactual soft answer credit for discrete state existence

## Status and purpose

CASM-X20T is preregistered before implementation or execution.

Frozen X20S result base:

`9d4ebc5805f11e7e6e208de006878508111e201e`

X20S established that the X20R hard-forward discrete-state failure is not repaired by removing, delaying, or ramping storage pressure. Even the zero-storage straight-through treatment is seen-incompetent on 6/6 seeds, while the canonical live-mask control remains exact. The remaining bounded frontier is therefore the estimator/objective/credit-assignment path by which final-answer loss reaches discrete record-existence decisions.

X20T tests one narrow mechanistic hypothesis:

> the binary-forward task path fails because once useful records are absent, the hard execution trajectory provides a poor or saturated answer-credit signal to the constructor; adding a same-target counterfactual soft-execution answer loss can provide usable credit to the same raw gates without revealing live-state labels.

No live-mask, cardinality, intermediate-state, hidden-register, semantic-role, address, or causality supervision is introduced.

## Frozen base substrate

Inherit unchanged from the frozen X20S/X20R/X20 substrate:

- 8 supplied candidate entities;
- version-correct temporal dependency graph;
- shared reverse-topological graph constructor;
- deterministic nonlearned candidate identity codes;
- local-equivariant gated executor;
- final-answer-only task target;
- train live cardinalities `{2,3,4}` only;
- unseen live cardinalities `{5,6}`;
- train depth 12;
- evaluation depths 12/24/48/96;
- batch size 128;
- 12,000 optimization steps;
- AdamW, cosine LR `2e-3 -> 2e-4`, weight decay `1e-4`, gradient clip `1.0`;
- hard evaluation threshold `g_soft >= 0.5`;
- no top-k, count repair, matching, rerouting, or hidden oracle;
- Python `3.11.16`;
- PyTorch CPU `2.14.0+cpu`.

Frozen X20 data, executor, constructor, X20R straight-through estimator, and X20S evaluator files must remain byte-identical to the X20S result base.

## Gate definitions

For the graph constructor raw logit `z`:

`g_soft = sigmoid(z)`

`g_hard = 1[g_soft >= 0.5]`

`g_ST` is the exact X20R binary-forward, identity-through-soft-backward estimator. Its forward value is exactly `g_hard` and its backward derivative to `g_soft` is identity.

Define two final-answer losses using the same model parameters, same batch, same executor, and same final target:

- `A_hard = NLL(execute(g_ST), final_answer)`
- `A_soft = NLL(execute(g_soft), final_answer)`

Define storage terms:

- `S_hard = mean(g_ST)`
- `S_soft = mean(g_soft)`

The counterfactual soft path uses no extra target. It evaluates the same final answer under the same raw gates but without thresholding during that auxiliary execution.

## Frozen regimes

All learned regimes start from bit-identical compatible parameters for each seed and see identical training batches.

1. `canonical_live_mask`
   - unchanged positive control.

2. `hard_only_st`
   - exact X20R/X20S immediate hard-forward treatment replication;
   - objective: `A_hard + 0.05 * S_hard`.

3. `soft_x20_replication`
   - exact continuous X20 learned-instantiation objective;
   - objective: `A_soft + 0.05 * S_soft`.
   - diagnostic continuous comparator only; by itself it cannot authorize the discrete-state successor.

4. `soft_credit_hard_storage`
   - graph-conditioned decisive surrogate-credit treatment;
   - task credit is continuous but storage pressure is evaluated on binary forward existence;
   - objective: `A_soft + 0.05 * S_hard`.

5. `dual_credit_hard_storage`
   - graph-conditioned hard task path plus equal-weight counterfactual soft task path;
   - objective: `0.5 * A_hard + 0.5 * A_soft + 0.05 * S_hard`.
   - the `0.5/0.5` average keeps the task-loss scale comparable to the single-answer-loss controls rather than doubling answer weight relative to storage.

6. `soft_credit_hard_storage_structure_blind`
   - parameter-identical structure-blind constructor;
   - objective identical to `soft_credit_hard_storage`;
   - dependency/version message propagation removed exactly as in X20/X20R.

No entropy, polarization, margin, gate-label, live-mask, cardinality, count-matching, or curriculum loss is permitted.

## Fresh seeds

Use six fresh paired seeds fixed before implementation:

- train `20261201`, eval `20261281`;
- train `20261202`, eval `20261282`;
- train `20261203`, eval `20261283`;
- train `20261204`, eval `20261284`;
- train `20261205`, eval `20261285`;
- train `20261206`, eval `20261286`.

No seed replacement, selective rerun, schedule adaptation, coefficient sweep, or post-hoc threshold change is permitted.

## Integrity requirements

Before scientific training, tests must establish:

1. branch ancestry includes frozen X20S result `9d4ebc5805f11e7e6e208de006878508111e201e`;
2. this preregistration commit predates all X20T implementation commits;
3. frozen X20/X20R/X20S data, model, estimator, evaluator, and inherited tests are byte-identical to the frozen base;
4. all learned X20T regimes have identical parameter counts and bit-identical compatible initial state for each seed;
5. graph-conditioned learned regimes have identical raw `g_soft` before optimization;
6. `hard_only_st` loss is numerically identical to frozen X20R immediate loss when parameters and batch match;
7. `soft_x20_replication` is numerically identical to frozen X20 soft training loss when parameters and batch match;
8. `soft_credit_hard_storage` uses exactly `A_soft + 0.05*S_hard`;
9. `dual_credit_hard_storage` uses exactly `0.5*A_hard + 0.5*A_soft + 0.05*S_hard`;
10. `S_hard` forward equals the raw `g_soft>=0.5` record fraction and retains the frozen straight-through gradient;
11. hard evaluation remains raw `g_soft>=0.5` with no repair;
12. nonzero `A_soft` gradient reaches graph-constructor parameters on falsifier batches where the hard answer path is saturated or poor;
13. learned regimes cannot read live mask, active cardinality, final target value as constructor input, hidden trajectory, or causality labels;
14. no learned per-candidate gate/ID table is introduced;
15. structure-blind treatment receives the same observed candidate/program tensors except dependency/version propagation;
16. all losses, gates, gradients, and states abort on non-finite values;
17. inherited positive-control and executor regressions remain exact through `n=2..6`, depth 96;
18. every result records exact Git SHA, Python version, PyTorch version, seeds, optimizer contract, objective formula identifiers, and evaluation contract.

## Frozen evaluation and classification

Evaluation is unchanged from X20R/X20S and is applied independently to every regime from raw `g_soft`.

### Positive-control validity

`canonical_live_mask` must satisfy every seed and `n=2..6`:

- hard answer-final >=99% every suite;
- hard deep step-state exactness >=95%;
- hard deep live-register accuracy >=99%.

Failure invalidates X20T.

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

### Strong unseen state-instantiation extension

A strong PASS for a graph-conditioned discrete treatment requires **every 6/6 seed** at unseen live cardinalities `n=5,6` to satisfy the same capability and state-selection thresholds after first passing seen competence 6/6. No averaging rescues a failed seed or cell.

### Partial unseen extension

If strong fails but a seen-competent treatment has every unseen capability cell with hard/raw-soft answer >=90%, deep step/live-register >=80%, and hard existence F1 >=0.80, classify partial and report the exact boundary. Partial does not authorize reuse/merge.

## Credit-assignment interpretation rules

### Counterfactual-credit repair PASS

A bounded credit-assignment repair claim is authorized only if:

1. `hard_only_st` fails the 6/6 strong criterion; and
2. at least one graph-conditioned credit treatment (`soft_credit_hard_storage` or `dual_credit_hard_storage`) passes full seen competence 6/6 and strong unseen extension 6/6; and
3. `soft_credit_hard_storage_structure_blind` does not independently satisfy the same 6/6 strong criterion.

This would support only the claim that same-target continuous counterfactual answer credit repairs the frozen hard-forward optimization failure under this controlled constructor/evaluator.

### Relative mechanism interpretation

- If both graph credit treatments pass strongly, soft counterfactual credit is sufficient; do not claim the hard+soft mixture is necessary.
- If only `dual_credit_hard_storage` passes, report that combined hard+soft credit is required relative to the preregistered surrogate-only comparator in this setup.
- If only `soft_credit_hard_storage` passes, report that adding the saturated hard task loss is harmful or unnecessary relative to the preregistered surrogate-only comparator; do not generalize beyond this setup.
- If `soft_x20_replication` passes strongly but both hard-storage credit treatments fail, report that continuous task/storage relaxation remains easier than robust discrete-resource training; do not advance.
- If the structure-blind credit treatment also passes strongly, treat graph dependence as unresolved/shortcut-prone and do not advance to reuse/merge from X20T.
- If no graph-conditioned credit treatment passes 6/6 seen competence, remain on the estimator/objective/credit-assignment boundary.

## Claim boundary

Even a strong X20T result establishes only learned hard-thresholded **instantiate/ignore decisions over supplied candidate records from a supplied temporal program graph**, trained from final-answer supervision plus explicit storage pressure with a same-target continuous surrogate-credit path.

It does not establish:

- entity discovery from raw language/perception;
- semantic role induction;
- identity reuse/merge;
- lifecycle deletion;
- cross-episode persistence;
- program/controller induction;
- verifier-guided repair.

## Successor boundary

Reuse/merge is authorized only if X20T is valid and at least one graph-conditioned discrete credit treatment passes full seen competence on 6/6 seeds and strong unseen `n=5,6` extension on 6/6 seeds **without** the structure-blind treatment also earning the same strong PASS.

Otherwise, freeze X20T and keep the frontier on discrete state-construction estimator/objective/credit assignment.