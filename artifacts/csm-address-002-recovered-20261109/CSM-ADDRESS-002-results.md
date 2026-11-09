# CSM-ADDRESS-002: controller supervision repairs SET and recall in a bounded task

Research update for Woyengi Intelligence Labs · 9 November 2026

**The same 18,294-parameter CSM that largely fails with answer-only training
learned the controlled SET-and-recall task perfectly in all three seeds when
given explicit controller supervision during training.** Evaluation used raw
tokens and the model's own decisions, with no supplied addresses or write masks.
This is evidence for a useful training intervention on this restricted task.
It does not qualify CSM as a general language backbone or establish an advantage
over equally informed conventional architectures.

All twelve fixed-budget fits and evaluations completed. A separate review agent
verified and replayed all **39,936 prediction records**, with no weight updates.
One truncated baseline prediction file was recovered by frozen-checkpoint replay
to its exact original recorded SHA-256; the incident and incomplete original
console remain documented below.

## 1. Why this experiment was the next step

The recovered CSM-FOUNDATION-001 experiment tested a learned record-rewriting
model on SET, COPY and ordered subtraction. Its full-model answer accuracy was
12.11% on its IID panel, and independent traces showed near-constant routes and
answers. Its small byte-language experiment also remained behind the conventional
controls. Adding more graph structure or model size would not identify why that
learning recipe failed.

The present experiment isolates an earlier requirement: can the model store an
assigned value under the right identity, preserve other values, overwrite an old
assignment, and answer a later question about the requested register?

In ordinary terms, imagine four labeled notebooks. Instructions put one of eight
values into each notebook, change one of those values, wait, and then ask about a
specific notebook. A successful model must keep the notebooks distinct and use
the requested identity. Repeating one answer or remembering only the last value
cannot pass the complete-world and counterfactual tests.

The source computational primitive remains the original eager CSM transition:

\[
M'_d=M_d+\epsilon_v\sigma(g)(z-M_d),
\]

with a learned destination `d`, binary value-edit decision `epsilon_v`, sigmoid
update gain, learned proposal `z`, and a readout from the updated destination.
The remaining record values are preserved by the selected-write operation. There
are eight allocated records of width 32, four operation codes, and pointer heads.
This experiment does not establish a benefit from the pointer or operator features.

## 2. What changed in training

Four conditions were registered before fitting:

| Condition | Training information | Evaluation information | Parameters |
|---|---|---|---:|
| Original CSM (`raw`) | Raw tokens and final answers | Raw tokens only | 18,294 |
| CSM with controller supervision (`supervised`) | Same tokens/answers plus destination, admission and update-gain targets | Raw tokens only; all decisions learned | 18,294 |
| Supplied-controller diagnostic | Correct destinations, write timing, unit gain and disabled link editing are supplied; proposal/readout are trained | Same privileged controller information | 18,294 |
| Conventional Transformer | Same raw tokens and final answers | Raw tokens only | 18,257 |

For the supervised condition, the objective was

\[
L=L_{answer}+L_{destination}+L_{admission}+0.25L_{gain}.
\]

Destination targets apply at the value token of a SET and at the final query
token. Admission targets identify which tokens should change stored values.
The gain target encourages complete overwrite at SET value positions. These
targets impose a mapping from register names to physical slots during training.
They are privileged supervision, not discovered semantic roles.

The supervised condition has the same forward computation and parameter count
as the original model. The added losses create no additional inference module.
Prefit tests changed every controller target and answer metadata while holding
the input fixed: the supervised forward output was unchanged. The supplied
controller was tested separately and is never counted as learned control.

The Transformer is a task-calibration reference, not an equally supervised
competitor to the supervised CSM. Matching parameters and answer exposures does
not match the amount of training information, FLOPs, state capacity or training
time. Its poor score under this budget does not establish a Transformer limit.

## 3. Data, budget and prospective screens

Each source world initializes four registers in random order, then assigns a
different value to one register. Four NOP tokens separate the final write from
`ask r ?`. There are 512 training worlds, each queried independently at all four
registers: 2,048 question presentations. Every question begins with fresh model
state and receives no prior gold answers.

| Held-out panel | Construction | What it tests |
|---|---|---|
| IID recall | 128 fresh worlds, four separate queries each | Correct identity and latest-value retrieval |
| Delay 16 / delay 64 | Same IID worlds with longer NOP intervals | Persistence beyond the training delay |
| Counterfactual | 64 families, eight variants per family, exactly one changed value token | Dependence on the relevant earlier assignment |
| Irrelevant writes | Four extra SETs to nonqueried registers | Preservation under unrelated changes |

Every panel has 512 questions and exactly balanced answer values. A constant
value scores 12.5%. Copying the last written value scores 35.35% IID, 12.5% on
counterfactuals and 13.48% after irrelevant writes. Counterfactuals deliberately
query a value written before the final overwrite, and all eight variants remain
in one held-out family. Counterfactual register frequencies are not balanced.

Register-renaming-canonicalized complete histories are disjoint across TRAIN,
IID and counterfactual panels. Counterfactual family signatures are excluded from
TRAIN and IID. This does not make every possible final semantic state disjoint.
Delay and irrelevant-write panels reuse IID worlds and are paired interventions,
not independent replications. Irrelevant-write group exactness combines four
query-specific perturbations of one source world.

Each of seeds 101, 202 and 303 received 256 AdamW updates, batch 32, learning rate
0.002, weight decay 0.01 and gradient clipping at 1.0. That is 8,192 answer
exposures per fit. All arms used the same sampled example-index schedule within
a seed; all three CSM conditions started from identical weights within a seed.
Final fixed-update checkpoints were used. No early stopping, checkpoint selection,
test-guided extension or retraining occurred.

The source, configuration and data were frozen in preregistration commit
`5bf1765`, after independent review and ten passing prefit tests. All twelve fits
closed before any trained-model held-out scoring. The registered continuation
screens required every seed to meet 95% IID, 90% complete counterfactual families,
95% delay16, 90% delay64 and 95% irrelevant-write accuracy. These are descriptive
screens for the next research decision, not population-level significance tests.

## 4. Results

All percentages below are means of all three registered seeds.

| Condition | IID recall | Counterfactual answers | All eight counterfactual answers correct | Delay 64 | Irrelevant writes |
|---|---:|---:|---:|---:|---:|
| Original CSM | 13.61% | 13.74% | 0.00% | 13.61% | 12.57% |
| CSM with controller supervision | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Supplied-controller diagnostic | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Transformer reference | 34.11% | 31.45% | 1.04% | 29.69% | 25.78% |

The supervised and supplied-controller conditions also achieved 100% on delay16
and on all four queries of every IID world, in **every seed**. Both passed every
registered continuation screen. The original CSM and Transformer failed those
screens. The Transformer correctly completed two counterfactual families in
seed 202, and none in the other two seeds.

### Per-seed IID results

| Seed | Original CSM | Supervised CSM | Supplied controller | Transformer |
|---|---:|---:|---:|---:|
| 101 | 12.50% | 100.00% | 100.00% | 35.35% |
| 202 | 12.50% | 100.00% | 100.00% | 38.28% |
| 303 | 15.82% | 100.00% | 100.00% | 28.71% |

The original condition predicted one value for every IID question in seeds 101
and 202. Seed 303 used two values and improved slightly, but completed no IID
world or counterfactual family. Simplifying the task alone did not repair the
answer-only learning recipe at this budget.

### Learned control was checked directly

On IID, the supervised model in every seed correctly selected all 3,072 scored
destination decisions: five writes and one query for each of 512 presentations.
It classified all 14,336 write/no-op positions correctly, admitted all 2,560 true
writes and made zero false writes. Every overwritten-register and retained-register
query was answered correctly. Canonical physical-slot agreement is meaningful
here because that mapping was explicitly supervised. It would not be a valid
failure criterion for an unsupervised model using a consistent permutation.

### The answers depend on stored values

Immediately before `ask`, the diagnostic either erased only record values or
rolled values by one slot. It preserved the question tokens, frontend history,
cursor, links and weights.

| Condition | Normal IID | Values erased | Values moved to other slots |
|---|---:|---:|---:|
| Original CSM | 13.61% | 12.50% | 12.50% |
| Supervised CSM | **100.00%** | **12.83%** | **15.23%** |
| Supplied controller | 100.00% | 12.50% | 14.58% |

The large degradation supports use of persistent values and their bindings on
this task. Perfect performance on all eight counterfactual values rules out a
constant-answer explanation. Perfect paired correctness after irrelevant writes
is stronger than mere agreement between predictions that could both be wrong.
These interventions do not establish general semantic understanding.

## 5. What the result establishes and leaves unresolved

**Supported:** this implementation can learn reliable token-driven SET, overwrite,
preservation and delayed recall in the fixed four-register/eight-value grammar
when it receives explicit controller credit during training. The new supervised
condition uses its own predicted controls at evaluation. The earlier negative
result was therefore not evidence that the update/readout machinery is incapable
of even this simpler task.

**The causal conclusion is narrower than "address learning is solved."** The
intervention simultaneously adds destination, admission and update-gain losses.
It does not identify which loss is necessary, which interactions matter, or
whether less privileged supervision would work. The supplied-controller arm
also changes multiple control choices and is only a capacity diagnostic.

**Still unestablished:** learning internal roles from answer supervision alone;
natural-language parsing; COPY after source overwrite; arithmetic or ordered
composition; useful pointer traversal; larger numbers of registers or values;
new identities or operation tokens; open-ended structure discovery; competitive
language quality; architectural novelty; and a favorable quality/cost frontier.

Three seeds at one small budget provide a repeatable controlled result, not a
generalization guarantee. The trained Transformer reference also failed the
strict screens. No claim of superiority over a properly calibrated or equally
supervised Transformer follows.

For the inference-priority research program, the useful engineering fact is that
the intervention adds training objectives without adding an inference module or
model parameters. No new latency benchmark was run, and no 12:1 inference/training
efficiency score is claimed. The old negative parallel-runtime result remains.

## 6. Formal and numerical boundaries

The recovered CSM-GRAPH-001 receipt reports 76 declared theorems and 273 audited
theorem constants. An independent source check matched all twelve recorded Lean
source hashes. Lean was not installed in this executor, so compilation was not
rerun here.

Those proofs establish conditional structural properties such as selected-write
frames, bounds, causality and schedule equivalence for abstract pure kernels.
They do not establish that learned addresses, proposals or readouts implement
the intended register semantics. `SEMANTIC_CONTRACT.md` makes the required
representation relation explicit:

\[
\mathcal R(M,\sigma)\iff\forall r,\quad
\operatorname{decode}(M[b(r)])=\sigma(r),
\]

where `b` is an injective register-to-slot binding. Correct destination, semantic
overwrite, preservation of other bound records and correct queried readout are
sufficient to preserve this relation. The note is a mathematical argument,
**not a newly compiled Lean theorem**. This experiment supplies finite empirical
tests of the behavior rather than a proof that training always establishes it.

The formal archive also preserves the existing numerical and backward limits:
dense floating-point masking has nonfinite/signed-zero boundaries; a scripted
serial control had None-versus-zero gradient discrepancies; and the measured
fork/join implementation was slower. This experiment retained unchanged eager
computation and checked exact forward and gradient-presence correspondence.

## 7. Verification and the preserved evidence incident

The prefit review independently parsed all 4,608 generated examples, checked
splits and counterfactuals, tested absence of inference metadata leakage, and
reran ten software checks. A prefreeze import-order defect in the adapter was
fixed before fitting; the correction and exact reviewed hashes are retained.

The postfit review used separately written audit code, imported the frozen model
definitions, independently recomputed metrics, and replayed every condition for
every checkpoint. It verified twelve checkpoint/optimizer pairs, 256 updates per
run, paired initialization and schedules, 39,936 saved predictions, 78 condition
metric rows and 54 paired comparison rows. Every predicted token reproduced.
The largest independently computed NLL discrepancy was 9.58e-7 nats, within the
registered audit tolerance of 5e-6. This is review-agent verification and replay
in the same environment, not external replication or independent retraining.

The first audit stopped on a real artifact-integrity mismatch: Transformer seed
303's saved predictions had only 2,029 of 2,560 rows. Its recorded complete-file
hash and summary remained available, and the original process console was also
incomplete. The cause is unknown. A separate replay from the unchanged checkpoint
reconstructed all 2,560 rows and reproduced the **exact originally recorded hash**,
`4e15a6e53494e472c2a7207e8995eee9f4799838a23cf38e4128f84cd5bf16ad`.
All surviving bytes were an exact prefix; all original metrics matched.

The damaged original and console, initial audit failure, replay code and recovery
receipts were retained. Canonical predictions were restored only after the exact
hash matched, and the complete audit then passed. The original console remains
incomplete; no complete original log is claimed. No weights, training inputs,
metrics or checkpoints were changed to repair the artifact.

Runtime: Python 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.3.5; three workers with one CPU
thread each. Concurrent fit times are execution receipts, not controlled speed
benchmarks. All twelve fits totaled 3,072 optimizer updates and 98,304 answer
exposures. No GPU or paid compute was used.

## 8. Next decision

Keep this as the first successful **supervised SET/recall reference**. The next
experiment should use fresh frozen data and ablate destination, admission and
gain supervision, keeping the successful full-supervision condition and an
answer-only control. That will identify the minimum useful internal credit.
After that, test reducing/removing such labels before claiming learned structural
discovery. COPY with later source overwrite is the next capability stage once
the controller-learning question has a sufficiently reliable reference.

Do not scale CSM or interpret this result as language competence yet. The immediate
research priority is learning the controller with less supplied structure while
preserving the demonstrated state behavior.

## 9. Archive and publication status

The recovered foundation and formal source commits are `a89ae4e` and `0e667ffd`.
Their saved packages contain all thirty original checkpoints and all 242 original
raw evidence files. The exact previously reported consolidation commit `168d44d`
was unavailable; the recovered source identities and archive hashes are recorded
instead. Recovery commit `6751beb` preserves both original ZIPs and a hash-checking,
non-overwriting restoration script. The present experiment adds twelve checkpoints.

Working branch: `research/csm-recovered-address-002-20261109`.
Preregistration: `5bf1765`. Public upload remains blocked: automatic approval review
did not accept recovered historical approvals as authorization in the current
trusted transcript. No alternate publication route or merge was attempted.
The complete new results and recovered archive are prepared for review and saving.

### Evidence navigation

- `PROTOCOL.md`, `prefit_manifest.json`, `config.json`: frozen design and source.
- `data/manifest.json`, `data/panels.json`: dataset definitions and bytes.
- `training_closed.json`, `runs/*/complete.json`: completion and provenance.
- `runs/`: checkpoints, optimizer/RNG state, updates and all predictions.
- `summary.json`: all per-seed results, means and continuation screens.
- `PREFIT_REVIEW.md`, `POSTFIT_AUDIT.md`, `POSTFIT_AUDIT.json`: reviews.
- `postfit_recovery/`: the preserved artifact incident and exact-hash recovery.
- `diagnostics/`: read-only diagnosis of the original failed checkpoints.
- `SEMANTIC_CONTRACT.md`: mathematical semantics and proof boundaries.

All quantitative findings in this report derive from these local experiment
records. No external performance or novelty claim is used to support the result.
