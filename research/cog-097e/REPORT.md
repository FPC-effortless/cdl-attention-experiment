# COG-ATTN-097e — Attention-free raw-observation action learning, no supplied answers

**Decision: NOT QUALIFIED.** Real training occurred, including online actions that received binary outcomes, followed by frozen test-time inference; the learned policy was nonetheless **less accurate, less selective, and slower** than a generic character-TF-IDF reader. This research tests a read-only subsystem, not a full COG-ATTN controller, no human-agent interaction and no emergent general intelligence.

## Setup

- Data: naturally authored CPython 3.13.5 standard-library docstrings, using their opening authored summary as the request and the rest of the authored prose as a memory record. This is a synthetic **task derived from authentic prose**, not real conversation. Identical bodies or query summaries removed across module-disjoint train/dev/test partitions.
- Training: 297 original documents; three incomplete archives per seed with unpredictable missing records. Every query was an authored summary and the agent chose retrieved passages from a whole-archive lexical index. Hidden world returned exactly one bit (`SELECT` success/failure) after its chosen action. No correct candidate, entity identity, semantic schema, source label or gold field passed to learner. Up to three action attempts; 891 requests per seed.
- Policy: generic word/character TF-IDF retrieval as part of *every* reader. Learned branch used trainable hashed word/subword EmbeddingBag and MLP with gradients into input representation; its reranker saw raw request/candidate prose and unsupervised lexical scores. `no_encoder` and shuffled-outcome arms isolate representational learning and causal feedback.
- Development: separate source modules, 48 archived and 13 missing requests; thresholds selected on binary outcome feedback with explicit >=15% development coverage, `correct - 2*wrong - .05*abstain`. No test-time feedback.
- Test: different source modules, 161 archived + 23 missing = **184 distinct requests**, same cases across seeds 1701, 1707, 1711. 552 model-seed decisions are correlated evaluations, **not 552 independent documents**.

## Actual results (3 seeds; 552 decisions per arm)

| Method | Forced top-1 correct | Accepted correct | Accepted wrong | Accepted missing-evidence wrong | Total accepted | Precision / coverage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Character TF-IDF | **216** | **78** | **36** | **0** | 114 | 68.4% / 20.7% |
| Word TF-IDF | 171 | 33 | 30 | 0 | 63 | 52.4% / 11.4% |
| Outcome-trained neural | 160 | 47 | 49 | 6 | 96 | 49.0% / 17.4% |
| Outcome-trained features only | 152 | 37 | 30 | 1 | 67 | 55.2% / 12.1% |
| Shuffled-outcome neural | 51 | 8 | 57 | 18 | 65 | 12.3% / 11.8% |

- The 3 learner seeds executed 2,673 train episodes, 5,788 attempted actions and 1,073 observed successes. Incomplete memory created genuine absent-document decisions. Outcome feedback was available only on those actions, never at frozen TEST.
- Separate candidate-coverage diagnostic, using the full real 161-document test archive: union of top-8 word and top-8 character TF-IDF retrieved the true passage for **135/161** queries. Word top-1 had 57/161, character top-1 had 72/161. In the 26 candidates-absent cases, no reranker restricted to this union could answer correctly regardless of learning.
- Median across seeds, character online local CPU ~3.7 ms/request including candidate retrieval; learned policy ~5.7 ms/request. Mean archive-build costs ~214 ms (character) versus ~211 ms (learned, since the shared word+char archive is the same design). Model training took 1.1–1.5 seconds/seed in the optimized version, *plus* about 4.7–5.3 seconds/seed for experience collection/indexing, and development calibration. These are not apples-to-apples total-cost victories; the learned arm is both slower online and less competent. CPU timings depend on host and warm caches.
- Exact frozen model checkpoints were saved as `097e-<seed>.pt`; an independently coded arithmetic audit checked all three seeds' per-case decisions against reloaded checkpoints for three comparator methods. Eight unit/invariant tests passed. Reproducing training on another machine remains open until hosted replay succeeds.

## Mechanism diagnosed

Gradient flow from binary action feedback into a continuous encoder **was implemented**. That solves a software plumbing problem but not the research problem. Reward arrives primarily for candidates that lexical similarity could already retrieve; the training signal is biased toward frequent lexical anchors. The learner does not discover representations for unseen semantic relationships, and it loses against a simple general-purpose character index. Adding a trainable reader to a stronger retrieval backbone can *degrade* decisions. The hallucination/unknown gate is learned only through a small development calibration; it did not make missing-reference errors vanish.

## Precommitted decision

Do **not** promote this as attention-equivalent language understanding, successful open-ended skill acquisition or efficient learned reasoning. The next **097f** test should change the experimental target, not tune on this sealed test: require a real reversible environment with non-text consequences and predeclared action permissions, raw observations and sequential outcomes, novel tasks after training, learned representation/world dynamics, no gold solver, and separate action-authority control. Keep clear baselines that receive the same interaction budget; do not weaken them to manufacture neural gains.

## Publication and provenance

`experiment.py`, `vendor_097c_experiment.py`, `PROTOCOL.md`, `test_experiment.py`, `audit.py`, `summarize.py`, `SUMMARY.json`, `AUDIT.json`, source hashes recorded in each per-seed result, and replay commands are required for source-level reconstruction. Full per-instance traces/checkpoints are additionally in the accompanying archive. Preserve negative results and source-dataset hashes on any public branch; public source alone is **not** equivalent to uploading all binary training artifacts.
