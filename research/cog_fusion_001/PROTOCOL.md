# COG-FUSION-001: prospective build and qualification

Date: 23 November 2026. Source base: `5d5f705e80ba3c77e303915f0293d87bba12af8d`.

## Question and boundary

Can a roughly one-million-active-parameter COG model interpret a continuous
history, update protected current records, retrieve selected current evidence,
and generate the correct characters through its actual language head, including
correction and irrelevant-edit pairs? Can that same executable system resume
without re-encoding historical input and avoid stale answers during known
pending changes?

This is a controlled lifecycle integration experiment. It does not qualify a
general LLM, establish arbitrary semantic correctness, compare against frontier
attention implementations, or demonstrate general speed superiority. The fixed
16-identity vocabulary, synthetic wording and short main training histories
limit the scope of every empirical conclusion. Long filler histories are a
separate extrapolation diagnostic, not trained long-horizon competence.

## Architecture and controls

- Native: selected 993,574-parameter Connected COG checkpoint, SHA256
  `d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c`.
- All six local affine blocks, embedding, normalization and original
  65-character head remain trainable. No frozen-feature cache is used.
- Shared continuous local state; an event-local GRU reads the same computed
  features and resets only at declared event boundaries.
- Learned typed write/query heads; protected exact executor with stable
  entity/relation identities. Exact two-hop resolution follows its actual first
  value. That typed traversal rule is deterministic, not a learned hop policy.
- Optional hybrid predicts two of 16 buckets and scores at most four actual
  candidate records per hop using learned compatible identity representations.
  Its second query consumes the actual first selected value. Index construction
  and 16-way codebook classification are additional work and must be counted.
- Up to two record values feed a learned controller and projection into the
  original 65-character logits. No answer choices, spelling constraints, label
  renderer, or gold answer length enter candidate generation. Stop on generated
  period or 12 generated characters.
- 1,077,834 stored parameters; 997,674 active; 80,160 disabled legacy-board
  parameters retained and explicitly counted. Fixed neural tensors: 10,592
  bytes per session, including retained inactive state.
- Matched arms: `memory` and `no_memory`. Same initial tensor values per seed,
  data order, all auxiliary supervision, optimizer, updates and architecture;
  no_memory zeros the factual evidence bridge throughout training/inference.
  This is an evidence-connection control, not an optimized GRU/Transformer.
- Acute removal from the trained memory candidate, hybrid reader, supplied
  reference evidence, and bounded recent-write cache are diagnostic arms.

## Training lock

Three new-interface/data-order seeds: 17, 29, 41. Both arms per seed. Each fit:

- 1,000 updates, four paired families/eight worlds per batch, fixed final
  checkpoint. No held-data checkpoint selection or extension.
- All five events, query and answer share carried state and a complete
  within-history backpropagation graph. Histories reset between independent
  training examples. The event accumulator resets at each event.
- First 250 updates use supervised typed writes and query identifiers for
  memory execution; the final 750 execute model-predicted writes and queries.
  Both phases supervise semantic heads. Only training uses this curriculum.
- AdamW, weight decay 0.01, native LR 0.00015, interface LR 0.003; cosine decay
  to 20% of initial LR; norm clipping at 1.0.
- Loss: event/query semantics + 0.5 route supervision + 2 answer-character CE
  + 0.2 paired first-character counterfactual/invariance loss. Every fourth
  update adds 0.2 TinyShakespeare TRAIN rehearsal CE on four 128-target windows.
- Hybrid route supervision compares 16 bucket classes and 32 logical keys per
  hop in training. Inference cannot reuse this full-key supervision.
- CPU only; one Torch thread per fit; 1,800 CPU-second cap per fit. A cap hit
  leaves an explicitly incomplete checkpoint, not a qualified fit.
- A five-update train-only calibration precedes this lock and is excluded from
  reported results. It measured about 0.7 CPU seconds/update after startup.

## Data and prevention of contamination

`DATA_MANIFEST.json` binds 1,024 TRAIN paired families, 64 DEV, 128 familiar TEST,
128 held-wording families and 32 longer filler-history families. Eight kinds:
direct, two-hop, correction, irrelevant, retraction, bridge, missing, cancelled.
Both sides always differ in text; identical alleged counterfactual pairs are
rejected. Canonical semantic worlds are deduplicated across splits, including
invariant-control histories. Fixed entity names are shared intentionally.
Fresh wording changes both general vocabulary and syntax, so it is not an
isolated syntactic-generalization test. Report answer-changing versus
invariant/unknown pair denominators separately. Do not pool seed replicas as
independent data points.

## Outcomes and gates fixed before main training

Engineering: all executable correctness tests pass, including protected writes,
explicit capacity exhaustion, actual multi-hop use, missing/deleted/recreated
dependency invalidation, fork identity, pending scopes, coherent transactions,
early idempotency, restart, RNG/input accounting, and crash-before/after-COMMIT.

Learned qualification requires each of the three memory fits to achieve:

1. At least 95% full generated-answer exact match on familiar and held wording.
2. At least 95% complete paired answers on both panels; report each kind and
   answer-changing/invariant split so averages cannot conceal a failure.
3. At least 95% whole-history exactness: every predicted operation, every
   post-event ledger, supporting read and final generated answer all correct.
4. Current relevant evidence changes cause the appropriate answer changes in
   fixed-state interventions; irrelevant edits preserve correct answers.
5. No more than 10% relative TinyShakespeare validation NLL degradation from
   the admitted local-only parent on fixed 64x128 validation targets. The full
   original native model is an additional reference, not an interchangeable
   local-only baseline.

Any failure retains a useful engineering/diagnostic result but blocks complete
learned-system qualification. A failed small run does not prove the architecture
cannot learn with more appropriate scale, data, training or objectives.

## Cost and reporting

Time full interpretation, lookup/index, neural generation, validation, writes,
transaction publication and restart separately and together. Benchmark timing
after training workers stop. Compare packet reuse with repeated resolution;
include its validation cost. Count processed new input characters, generated
characters, historical replay characters, actual candidates, state tensor bytes
and database/archive bytes separately. Growing durable history must not be
described as fixed total memory.

The inference-weighted 12:1 preference is applied only after comparable quality
has passed. No quality failure earns an efficiency win by being faster.
