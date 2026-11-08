# VB-001: missing versioned-memory baselines, reconstructed task

Prospective protocol, 2026-11-08. Freeze this file, source and integrity tests in
git before any optimization step. Base: LRC V4 integration branch b50e1615.

## Question and limitation

Can a tiny causal Transformer and a two-hop end-to-end memory network learn
versioned, dependent two-hop queries, using answer labels alone or identical
supporting-record labels? The source/checkpoints underlying the reported newer
LRC 100% versioned-memory result were not located. This is a **new reconstructed
experiment**, not its exact replication. No LRC comparator is invented. Results
cannot establish either advantage or disadvantage relative to that absent run.

## Input and task

Four named entities, four locations, two chronological eras. Each era contains
one partner and one location statement for each entity (eight statements),
shuffled within the era. Sixteen records total. Natural-language fixed templates
are tokenized by a finite, known word vocabulary. **Sentence boundaries are
provided**, and each sentence is encoded by a shared architectural design:
learned word embeddings, learned elementwise position weights, sum and layer
normalization. This is a structured template test, not raw byte language modeling.

A question asks where an entity's partner lived in a specified era. The model
must retrieve the partner and then that person's location in the requested
snapshot; questions cover both historical and latest eras. Two support occurrence
indices identify those statements. No entity identifier, parsed key, gold index,
or answer is passed separately into model.forward. Names and era words appear in
text. All models see the same sentence boundaries and chronology.

Examples come in counterfactual quartets: question, partner relation, and every
irrelevant fact remain identical; only the correct location value cycles through
all four labels. A deterministic query-only or memory-off model is exactly 25%
accurate. All-four-correct quartet rate tests sensitivity to the relevant fact.

For each entity, the next cyclic entity is excluded as the *queried* partner in
training. IID test uses allowed pairs; compositional test uses these excluded
queried pairs. All names, locations and individual fact types occur in training.
Excluded pairs may occur as nonqueried facts. This is held-out question/path
composition, not unseen vocabulary or globally withheld knowledge.

## Conditions, capacity and supervision

- Transformer: two causal pre-norm, one-head attention layers; width 32; FF width
  64. Operates on 16 sentence vectors plus the question vector. Query cannot
  attend itself. Learned occurrence positions; 19,076 trainable parameters.
- Memory network: two soft attention hops, width 51, untied key/value/state
  projections, shared sentence encoder, learned occurrence positions, residual
  state combination and layer norm; 19,129 parameters (0.28% larger). An adaptation
  of end-to-end memory networks, not an exact reproduction of the original paper.
- Each architecture has answer-only and answer-plus-support conditions. Support
  loss is the mean negative log probability of support 1 at hop/layer 1 and support
  2 at hop/layer 2, weight 1.0. Exactly the same labels are used. Labels supervise
  probabilities, never force test reads. Both conditions retain all 16 candidates.
- Query-only trained control: same sentence encoder design, width 32, two-layer
  classifier; 3,620 parameters. This is a negative control, not capacity-matched.

Reference: Sukhbaatar et al., End-To-End Memory Networks (2015),
https://arxiv.org/abs/1503.08895 . Transformer is a small custom reference
implementation; this is not a production throughput comparison.

## Fixed optimization and data

Three seeds: 301, 302, 303. Five conditions each, 15 independent fits. All train
from scratch. AdamW, learning rate .002, weight decay zero, gradient norm clip 1,
2,000 updates, batch size 32, no dropout/scheduler/early stop. Fixed 8,192-example
training set (2,048 quartets), generator seed 71001; sample with replacement.
Within each seed every condition sees identical minibatch indices in identical
order (generator seed 100000 + model seed). No validation tuning or budget
extension after seeing results. Capacity was selected by parameter counts only.

All 15 fits must finish before test generation/evaluation. IID and held-out-pair
tests: 256 quartets / 1,024 examples each; seeds 71002 and 71003. Verify no exact
complete input appears in training and either test. Final checkpoint only;
no best-seed selection. Record all seeds including failed/collapsed fits.

## Measurements and diagnostics

Primary: answer accuracy and all-four-correct quartet rate, each seed, each test.
Secondary: natural-log answer NLL; argmax support accuracy at each hop and jointly;
old versus latest snapshot accuracy. Train logs contain only sampled train loss.

Predeclared test ablations: zero all record token IDs (memory-off); replace soft
query reads by argmax reads at both hops. For Transformer, only the question row
is hardened; memory-record contextualization stays soft/causal. Thus hardening is
a within-model sensitivity diagnostic, not identical hard-index execution across
architectures. Record IDs cannot be returned from the network to an external
index in these baselines. No claim of sublinear retrieval or learned LRC wiring.

Record whole-fit CPU wall time. After all fits, time complete batch-one forward
passes (including sentence encoding), one CPU thread, interleaved model order,
30 repeats after warmup, 32 calls per timed block. Report medians and raw times.
Reference CPU classification latency is not optimized autoregressive throughput.
Do not score a 12:1 cost objective without a specified training amortization and
an agreed competence threshold.

## Integrity and evidence

Before fitting: tests for snapshot semantics, twin interventions, balanced
memory-off bound, deterministic generation, support-loss gradients, and no label
input to forward. After fitting: replay checkpoint predictions and confirm
source/data/minibatch hashes. Save every checkpoint, optimizer and RNG state,
all predictions, environment, raw timings and receipts. A local prospective git
commit records protocol and source; it is not an independent preregistration.
Treat quartet observations as dependent; no binomial p-values using 1,024
independent observations. Three seeds establish descriptive replication only.

No paid compute, no deployment, no merge. Preserve prior LRC implementation.
