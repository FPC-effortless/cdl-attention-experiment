# COG-WEIGHT-019 — Frozen-activation recoverability diagnosis

Date: 2026-11-22. Status: **completed diagnostic, no edits approved**. This work does not claim self-editing, general reasoning or full self-attention replacement.

## Question

Do the existing learned COG-WEIGHT-016 representations contain linearly accessible information identifying whether a two-document candidate supports a query, including under novel grammar and changing event order?

## Method

- Load the three original saved 016 selective-plasticity rankers (seeds 1701, 1707, 1711) with frozen pretrained COG-097e embedding weights.
- Extract the **actual 3×3 query/document slot cosine similarity matrix** and each candidate's original scalar logit. Flatten to 10 real-valued features per candidate.
- Fit a separate logistic-regression probe for binary candidate support on 017 *training* data, using gold final outcomes (candidate 0 / candidate 1 / neither) as probe targets. Select inverse regularization C only on 017 *development* data (candidate AUC). No probe sees any test labels during fitting/selection. No saved COG parameters updated.
- Evaluate the same probes on older withheld benchmark groups and newly generated test groups retained from experiments 017 and 018B. These groups were evaluated previously in earlier cycles; thus this is a post-hoc diagnostic **not an untouched external replication**. Three seed evaluations reuse much of the same synthetic test material and are correlated.
- Report both candidate-support AUC and **query-level answerability AUC** (maximum of two candidate probabilities vs ground-truth presence). This does not claim calibrated acceptance or improvement of the original ranker.

## Results

| Test cohort | Mean answerability AUC across 3 seeds | Interpretation |
|---|---:|---|
| old_matched_test | 0.998 | Familiar ordered relationships linearly recoverable |
| old_roles_test | 0.992 | Familiar explicit role reversal linearly recoverable |
| old_history_test | 0.771 | Weaker event-history support signal |
| history_long_fresh | 0.774 | Long histories remain only moderately discriminable |
| old_unseen_test | 0.661 | Earlier unfamiliar wording transfers poorly |
| old_paraphrase_test | 0.806 | Some paraphrase information retained |
| unseen_fresh | 0.541 | Novel syntax near chance |
| history_ood_fresh | 0.605 | New history wording weak |
| new_syntax (018B) | 0.529 | Unseen support phrasing near chance |
| new_history (018B) | 0.779 | Known temporal grammar generalizes better |
| new_negation (018B) | 0.912 | Direct negation cases often readable by a linear probe; acceptance calibration remains unproven |

Per-seed AUCs, candidate-level AUCs, top-1 comparisons, cohort hashes, and model checkpoint SHA-256 hashes: `RESULTS_019.json` and `SUMMARY_019.csv`.

## What the study *does* and *does not* show

**Supports:** Frozen ranker internals contain a useful readout for some already learned relational patterns. The existing original verifier/calibration functions do not reliably expose that signal on unfamiliar syntax. Direct edits to gating or small heads in 018 previously failed all promotion gates.

**Cannot show:** That new semantic relations are wholly absent from activations. A nonlinear decoder with higher capacity or alternative training could extract them; this study evaluates only a specific 10-feature linear probe. It cannot establish causal importance of individual neurons; for that use interventions and matched activation patching. It is not a zero-label verifier, since probes use outcome-labelled training rows.

**Unresolved:** Representation-level compositionality under unseen language; safe learning of new predicates; multi-event contradictory updates; no-answer calibration; full end-to-end inference cost and external text corpora.

## Decision

**Reject promotion of any new model**, because no new trained replacement was produced and unsupported-answer behavior remains weak in novel syntax. Retain all historical original/failed checkpoints. Next use paired minimal counterfactuals and activation interventions, with frozen protocols, to identify the earliest lost information stage; compare fixed embeddings vs pooled slots vs similarity maps, then design and train a role-sensitive backbone only if needed.

## Replay

Place the archived COG_WEIGHT_016_Selective_Plasticity_20261122, COG_WEIGHT_017_Relationship_Verification_20261122 and COG_WEIGHT_018_Circuit_Repair_20261122 directories at a common root beside COG_WEIGHT_019_Activation_Probes_20261122. Run:

```bash
export COG_WEIGHT_ROOT="$PWD"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python COG_WEIGHT_019_Activation_Probes_20261122/probe_019.py
```

Dependency versions in measurement environment: Python 3.13; torch 2.10.0+cpu; numpy 2.3.5; scikit-learn 1.8.0.