# Next qualification protocol: COG-ATTN without supplied solutions

**Date:** 2026-11-21. This protocol was written **after** inspecting the 097c diagnostic and is **not a preregistration for its outcomes**. It specifies gates for new prospective work.

## Disallowed learner inputs

- Golden entity names/IDs, target slots, relation labels, action labels for intended requests, answer candidates, resolved coreference graphs and handcrafted semantic parse results.
- Oracle reference paths, predetermined action macros, demonstrations tailored to test utterances, exact held-out paraphrase support, true verifier answers or top-k lists containing the answer by construction.
- Test-set annotations, labels, future turn content, hidden environmental rules or answer leakage from scorers to policies.
- Pretrained semantic model presented as if learned from raw experience; any pretrained comparator must be reported as such and budget matched where possible.

## Permitted physical interface

A world must expose *what the agent can observe or do*, not what the observation means. Raw dialogue and/or raw pixel/byte observations, a generic primitive action API, timestamps, success/failure and sparse environment feedback are allowed. An action API is not permission to disclose desired action or goal. An agent must learn mapping, identity, schema, retrieval relevance, and procedures from its own feedback or unlabeled observations. The evaluation checker remains isolated.

## Prospective datasets / worlds

1. MultiWOZ 2.2: official disjoint train/dev/test dialogues, authentic raw user and system utterances to learner, external gold dialogue states only inside a sealed scorer. Do not feed state ontologies or gold slot spans to model. Separate coreference, corrections, unseen domain and silent-state tests. Publishing corpus copies requires provenance/licence handling. Source: https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2.
2. A separate unstructured workflow corpus with free-form user instructions and observable issue-state effects, no synthetic gold operator labels at learning time. Execution remains sandboxed or read-only until independent authorization.
3. Held-out autonomous physical puzzles where a raw observation and generic action list are available; no solution planner in learner process. Reward depends on terminal success, legality and resource consumption; test on heldout world families and changing goals.

Do not substitute generated prompts or function-name labels for real language qualification. If authentic data cannot be obtained, **report data access blocked** rather than simulate a passing result.

## Ablations and controls

- Same-task, same-input, same-data, same-training-budget BM25, standard dense Transformer, and no-memory controls. Distinguish pretrained baselines from models learned from scratch.
- Separate memory retrieval recall, answer correctness, calibration/unknown behavior, semantic miscommit, unauthorized action, stale-state violation and skill transfer.
- Fix maximum candidate probes k, memory budget, observation count, training updates, action budget and wall time before heldout evaluation. Include learned representations and any verifier work in end-to-end cost.
- Audit raw-state history preservation, source revision proofs, time-dependent changes, missing evidence and counterfactual wrong actions. Unsafe actions forbidden; verifier denial is not semantic correctness.
- Hold out entire task/domain/utterance paraphrase families and sources, not just entity surface names. Prevent source-derived data duplication and indirect leakage.

## Promotion gates (hypotheses, not observed)

Pre-register numeric thresholds on dev only, then score heldout exactly once. Capability promotion requires heldout task competence at least as high as the strongest fair baseline, improved out-of-distribution generalization or materially lower matched-competence all-in latency, and no degraded authorization or provenance invariants. If performance loses, preserve the negative result. Archive model, datasets manifest, optimizer state, seeds, per-case traces, reference implementation and exact commands; independent verifier should reproduce scoring on a separate machine.
