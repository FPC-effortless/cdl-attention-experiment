# Binding separation and recovery from biased training

This cycle tests two separable hypotheses after the prior grounding failure. Research Lab OS froze four protocols, TrainLab recorded 24 jobs, and the Veritas ecosystem reference comparison passed reverification. The custom experiments use exact domain oracles and review the saved model outputs independently of the runtime scoring code. All experiments use constructed inventory instructions or finite programs.

## Separate role grounding from verb induction

The local model masks verb identity and learns source and destination from context by marginalizing compatible operators. It trains from random initialization on 4,096 state transitions. It learns meanings for four extra spellings from 96 separate support transitions without changing the role model. Meanings are permuted by seed and the exact supported spellings are declared. The independently initialized joint GRU and Transformer controls use the same training commands and support examples. They fit both factors jointly with familiar replay during support adaptation. The full-sequence ablation shares the factorization and support procedure with the local model.

| 1,536 examples per cohort | Local modular | Full-sequence modular | Joint GRU | Joint Transformer |
| --- | ---: | ---: | ---: | ---: |
| Familiar | 100.0% | 100.0% | 100.0% | 100.0% |
| Novel verbs in support format | 100.0% | 100.0% | 100.0% | 100.0% |
| New layouts, known verbs | 100.0% | 92.8% | 19.7% | 5.4% |
| New layouts and novel verbs | 100.0% | 92.7% | 17.8% | 5.1% |
| New implicit role wording and novel verbs | 20.6% | 0.0% | 5.6% | 8.3% |

The joint-layout test recombines familiar role phrases in unseen orders with different filler text and verb placement. This compares the modular package with the joint learners on a shared dataset. The package changes both the role encoder and the procedure for learning supported words. Replacing local windows with a sequence encoder loses 112 joint actions while still transferring much more than the joint controls. The local model inferred the novel operation on every implicit test command. Most of those exact predictions were wrong because argument roles failed. The local system met the predeclared joint-layout accuracy and confidence coverage gate. The independent Transformer failed the common competence gate, so this cycle omits speed comparisons. CUDA was unavailable.

The initial local model also admitted 512 of 1,536 ambiguous commands at confidence .80. A post-hoc two-model agreement rule accepted 1,207 of 1,536 joint-layout examples (78.6%), all correctly, and admitted zero ambiguous and zero implicit examples in this replay. It uses the same evidence and cannot guarantee independence or detect shared errors. The optional gate implementation is in recipe/admission.py. It is an abstention policy, and its measured ambiguity result needs fresh qualification.

### Held-out wording follow-up

With new seeds, half the familiar training examples were replaced by six authored plain-language patterns. The control uses the same local model, sample count and update budget. Both receive the same original novel-verb support. Fresh clauses reusing some role vocabulary improved from 490/1,536 to 501/1,536 exact (31.9% to 32.6%). Distant wording stayed around 11.2% (172 versus 173 correct). Increasing the paraphrase curriculum did not reach the 90% gate on either fresh cohort. Confidence became worse on distant wording: admitted false actions rose from 154 to 530 at threshold .80. Those observations make confidence a poor substitute for explicit clarification here.

## Recover operations from biased training

Given eight training observations, fit a bounded program and its measurement offset jointly. Select disjoint queries by disagreement and only admit when survivors predict the same results on unobserved cells. The true answer channel is independent in the recovery condition. The compromised channel echoes the original measurement. Test cells never participate in selection, training or validation.

| Scenario | Correct with restricted model | Correct with expanded nuisance inference | Incorrect admissions, expanded |
| --- | ---: | ---: | ---: |
| Constant offset, four independent queries | 50/240 | 240/240 | 0 |
| Constant offset, four compromised queries | 50/240 | 50/240 | 75 |
| Input-dependent offset, four independent queries under a constant-offset model | 1/240 | 2/240 | 0 |
| Piecewise offset, one independent query on fresh seeds | 17/240 | 219/240 | 0 |
| Piecewise offset, two independent queries on fresh seeds | 17/240 | 240/240 | 0 |
| Checkerboard offset, two independent queries on fresh seeds | 0/240 | 2/240 | 0 |

In the initial constant-offset suite, no anchor recovered 99/240 cases, one disagreement-selected anchor recovered 237/240, and two recovered all 240. Random selection gave 238/240 at both one and two labels and also reached 240/240 by four. Active selection had no uniform advantage. With four compromised queries, 75 wrong rules were still admitted. Both arms abstained on all tested random tables through eight independent queries.

The fresh piecewise suite allows two unknown offsets separated by one unknown threshold on either input axis. It recovered all 240 at two queries, before using the two optional blind anchors. The constant-offset comparator recovered 17. On the out-of-family checkerboard, the extended model admitted three wrong rules after one query. Two-query inference prevented all three, and the two blind anchors also blocked all three at the one-query budget. This is bias recovery within a supplied nuisance class. The checkerboard control exposes the limits of extending the class.

## Evidence and replay

The same-host audits reconstructed effects and hypotheses with separate arithmetic and checked TrainLab receipts. The frozen original and wording-follow-up suites contain 52,224 saved grounding predictions. The bias audits check 25,920 policy, budget and fixture outcomes. The agreement result is an exploratory replay of the original predictions. No trained model or test set was published externally.

Use Python 3.12, PyTorch 2.15 CPU, NumPy 2.3.5 and the four Veritas ecosystem packages at commit eec439492407b0aa26922c23b5ae40d95289f4eb. Start in a fresh directory with the supplied source layout and run:

    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python audit.py > audit.log
    python launch_piecewise.py
    python launch_implicit.py
    python audit_followups.py > audit-followups.log
    python agreement_check.py
    veritas-ecosystem verify "$PWD/ecosystem"

The package preserves the code snapshots, saved checkpoints and predictions, individual intervention traces, reports, manifests and lab database. Open-ended generation, unlabeled verb discovery beyond the supplied spellings, cross-host execution and accelerator performance remain beyond this cycle.
