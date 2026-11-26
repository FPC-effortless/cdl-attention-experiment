# MEMORY-INTEGRATION-003 — integrated memory and cost gates

## Authorized outcome and baseline

The requested work is to (1) make relevant accepted facts causally affect actual
model answers while irrelevant changes preserve them, (2) qualify the complete
learned memory path on whole histories and counterfactual pairs while preserving
familiar accuracy, and (3) reduce measured retrieval and persistence overhead
without weakening correctness. Implementation, bounded local CPU experiments,
and an update to the existing review branch are in scope. Paid compute,
deployment, merging, and claims of unrestricted lifelong memory are not.

The baseline is `ffb842adc176496d8838b774d0fe412aec5d1d20`, on
`feat/durable-memory-001`. MEMORY-VALIDATION-002 supplied the previous correctness
and overhead evidence. Its raw artifacts are retained unchanged. The prior
`PersistentDecoder` bridge supports the repository's reference Decoder and does
not consume accepted facts. This experiment must not relabel those untrained
engineering fixtures as the user's trained model.

## Two distinct model surfaces

1. **Trained factual experiment.** Admit the user's COG_CONNECTED_1M checkpoint
   from its supplied test kit and research archive. Verify archive manifests,
   checkpoint bytes, tokenizer and architecture before use. Freeze all 993,574
   native parameters. A small explicitly named factual adapter consumes the
   native post-board character features. Learned event, query and answer
   decisions connect to the durable typed archive. Exact schema validation and
   link traversal are infrastructure and must be disclosed as such. The native
   latent slots have no externally assigned semantic identities; no diagnostic
   route/commit override may supply gold labels at inference.
2. **Reference decoder engineering experiment.** Preserve all existing strict
   identity, full tensor-byte checking, receipt, erasure, state-codec and durable
   acknowledgment invariants. Its seeded untrained ~1M fixtures measure overhead
   only. A faster reference persistence wrapper is not evidence that the trained
   Connected COG factual mode has the same cost or language quality.

The selected native checkpoint was chosen in its prior research by validation,
not by this experiment. The original validation split is therefore a regression
check, not a fresh generalization test. Reproduce its full 55,769-target reset-128
validation, and verify that the frozen native path/weights remain unchanged after
the adapter is attached/trained. Keep private checkpoints and original corpora
out of the public repository.

## Learned qualification: prospective requirements

The detailed ontology, generator, lineage splits, hashes, fixed seeds, budgets,
and objective must be pinned in `learned/` before a scored fit. TRAIN controls
fitting; DEV may diagnose competence under the prespecified rule. Fresh TEST
wording families are scored after the recipe and checkpoints are frozen. Earlier
ANT/PULSE test families are exposed and cannot be called fresh here.

Compare familiar-only training with matched paraphrase/role-value-swap training
using the same initial backbone, head seeds, update count, batch budget and
evaluation histories. Report all three head seeds on this one selected native
backbone; these are not three independent pretrained backbones. Any train-only
budget calibration must set a common budget before scored fits. Planned combined
head-fitting CPU cap: 600 seconds; native feature extraction CPU cap: 900 seconds.
Account separately for preprocessing, cache I/O, model fitting, scoring and
durable serving. Do not hide inference or feature preparation as free training.

The principal endpoint requires **every event write decision across both complete
histories and both counterfactual answers** to be correct. Report first-error
position, operation/commit/address/value failures, final-ledger correctness,
required-chain correctness, answer-pair correctness, history length and wording
family separately. Invalid typed proposals count as failures, not oracle repairs.
An exact reader cannot rescue an incorrectly interpreted event.

Familiar DEV competence is a gate, with a target of at least 95% complete-history
and paired-answer correctness. The candidate must preserve familiar TEST accuracy
against its matched control; report paired outcomes and uncertainty, not just a
mean gain on unfamiliar text. A failure of basic familiar competence means this
bounded recipe is unqualified; it does not prove an architecture impossible.
Do not tune repeatedly on fresh TEST or substitute better internal head scores.

## Causal and integration checks

- For the same actual learned query/answer path, intervene on an accepted fact
  in its real read chain. Require the appropriate answer change for both values.
- Change a fact outside that chain. Require identical answers (and report the
  actual read dependencies). Test source deletion, stable absence, recreation,
  cross-session reopen and fresh-process reads as separate cases.
- The writer sees raw event text only. Query routing sees raw query text only.
  Inference has no generator/scorer labels, oracle write addresses, gold routes
  or gold answer fields. Learned operation receipts preserve retry decisions;
  a stale write cannot silently rebase onto a newer revision.
- An answer may acknowledge only after all actual fact revisions and source
  dependencies are revalidated in its durable transaction. Missing keys use
  revision zero; erased keys retain a monotonic tombstone revision. Corrections,
  insertions and erasure during inference must invalidate stale responses.
- Include archive-off and native-feature ablations. They establish whether this
  path uses the archive/features; they do not establish superiority to a simpler
  encoder, an open-vocabulary language-model gain, or native slot interpretability.
- Add independent root-level checks using real fitted heads after the interface
  is frozen. Deterministic stubs may test transaction faults but may not be
  reported as learned semantic accuracy.

## Serving and retrieval checks

`retrieval/PROTOCOL.md` pins the exact ranking equivalence and scope/erasure gates,
the existing 50k scale workload, and the stronger same-database 50k single-scope
paired comparison including Unicode. The first candidate removes redundant ASCII
normalization without truncating candidates or changing ranking semantics.

`serving/PROTOCOL.md` pins full-byte compatibility, adversarial identity checks,
old-v2 checkpoint resumption, and baseline/candidate reference-decoder timing.
Any confirmation run prompted by host contention must be prospectively specified,
retain the initial samples and run in a coordinated quiet CPU window. Report
complete durable calls as well as components. Do not equate asynchronous returns
or omitted safety work with a correctness-preserving improvement.

After changes freeze, run the complete memory package suite and the previous
fresh-process/system gate. Record the exact source tree and any optional private
dependency skip. CI must be checked on the final published head. A green storage
test does not waive a failed learned-memory qualification gate.

## Reporting boundaries

Separate implementation status, measured correctness, learned qualification and
cost. Retain failures and raw denominators. Synthetic bounded categorical memory
can falsify defects and establish specific behavior; it cannot establish lifetime
capacity, arbitrary fact extraction, free-form conversation quality or physical
power-loss durability. Report unresolved limits as unresolved.
