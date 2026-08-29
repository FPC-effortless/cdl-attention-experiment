# CASM v0.2 candidate — selective episodic writes

## Status

Prototype only. Do not treat this branch as an improvement over v0.1 yet.

The current best validated routing baseline is **compression-trained Q/K**: compression-derived predictive utility supervises ordinary Q/K scores during training, while inference uses the cheap Q/K read path without the pairwise compression-score MLP.

## Why change writes rather than reads

CASM v0 showed that an always-written memory path could become overconfident and brittle. CASM v0.1 corrected the read/injection side and established that retrieved memory is causally useful. The remaining question is whether every observed chunk deserves an episodic write.

v0.2 therefore preserves the v0.1 compression-trained Q/K read path and changes only episodic memory management.

## Mechanism

For each completed chunk, a causal write predictor observes:

- the current chunk summary;
- within-chunk observed prediction surprise (excluding the unseen boundary target);
- novelty relative to existing state/memory.

It predicts a write probability. A separate erase network predicts per-slot erase strengths conditioned on the new memory and each existing episodic slot. Erase and write are deliberately decoupled.

The episodic bank is a differentiable gated shift register:

- write=1 recovers the old append/shift behavior;
- write=0 preserves the existing memory bank;
- intermediate values softly interpolate between the two.

A memory-strength prior is included in Q/K routing so weakly written memories are less likely to dominate retrieval.

## Future-verified write supervision

During training only, the model measures whether the candidate new memory reduces the learned description length of actual future bytes. That detached future utility becomes an auxiliary target for the causal write predictor. At inference the gate has no future access; it predicts utility from current surprise, novelty and hidden state.

## Cold start

The implementation exposes `write_force`, allowing training to begin near the v0.1 full-write regime and hand control to the learned write policy gradually. This is analogous to the cold-start compression-teacher handoff already used for routing.

## Current smoke result

A same-initialization 100-step local pilot did **not** beat compression-trained Q/K:

- compression-trained Q/K hard answer NLL: ~2.7234
- selective-write candidate hard answer NLL: ~2.7503

The candidate is therefore not being scaled yet. The result is intentionally recorded as a negative early finding rather than tuned away.

## Next gate

1. Complete independent-seed replication of compression-trained Q/K.
2. Inspect whether gains remain concentrated in state tracking / graph structure.
3. Only then revisit selective writes, preferably with a simpler ablation sequence:
   - learned write only, no erase;
   - then separate erase;
   - then experience-window write features.

The purpose is to isolate which memory-management operation earns its compute instead of stacking multiple mechanisms at once.
