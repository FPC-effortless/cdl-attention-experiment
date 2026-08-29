# CASM-P — latent process supervision

## Hypothesis

CASM-R established that parameter-shared recurrence can improve answer likelihood without improving true solve rate. CASM-P tests whether the missing ingredient is **dense credit assignment over the latent computation trajectory**.

The deployed architecture is unchanged: a ~1.35M-parameter byte-level causal model with local GQA, compressed episodic memory, persistent recurrent state, and three shared Q/K memory-reasoning iterations.

## Critical benchmark correction

Legacy state-tracking data sampled an initial location for every object but omitted those initial locations from the prompt. If the queried object was never moved, the answer was unknowable. CASM-P explicitly serializes the initial state. All process traces are deterministic functions of visible input.

## Variants

1. `final-only-3step`
   - same corrected one-episode memory scope;
   - three shared recurrent reasoning iterations;
   - final language/answer supervision only.

2. `process-3step`
   - identical deployed core and initialization;
   - same final objective;
   - plus a training-only process head supervising each recurrent step against verifier-generated intermediate computation state.

The process head is discarded at inference. Deployed parameter count and inference path are therefore identical.

## Verifier-derived process traces

The targets are not model-generated chain-of-thought.

- associative recall: query identity -> relevant fact -> answer;
- state tracking: queried object's verified state after three event checkpoints;
- arithmetic: deterministic partial results;
- rule induction: inferred transformation -> instantiated rule -> answer;
- graph reachability: depth-limited reachable sets -> final reachable set/answer;
- reverse/copy: progressively verified output prefixes.

Intermediate text is encoded to a fixed order-sensitive continuous code. A training-only decoder aligns post-retrieval recurrent latent states to those codes with cosine loss.

## Causality guard

The supervised latent is taken at the final byte of the literal `answer ` marker. Causal local attention cannot see any gold answer byte at that position. Memory candidates are only those written by preceding chunks.

## Primary metric

**True autoregressive exact solve rate**, without gold answer length or teacher-forced answer bytes.

Secondary metrics:

- corrected hard-task answer NLL;
- answer-byte accuracy;
- exact-balanced graph yes/no accuracy by class;
- long-horizon state and associative performance;
- process alignment cosine;
- training and inference cost.

## Acceptance rule

CASM-P is promoted only if process supervision improves autoregressive exact accuracy over the paired final-only model on verifier-complete held-out tasks. A likelihood-only improvement is not sufficient.

## Falsifiers

Reject this form of process supervision if:

- process alignment improves but solve rate does not;
- benefits occur only in teacher-forced metrics;
- graph accuracy comes from single-class collapse;
- improvements disappear on longer state/associative contexts;
- process supervision harms heterogeneous-task generalization enough to offset structural gains.

## Relation to current work

The experiment is motivated by recurrent-depth latent reasoning and by 2026 looped-Transformer results showing that direct supervision of intermediate latent reasoning states can be necessary to convert recurrence into task reasoning. It also parallels the project's verified-continuation research: intermediate states are generated and checked by an independent executable task model rather than accepted as unconstrained hidden computation.

If this auxiliary-code formulation helps only partially, the next controlled ablation is direct LM-head supervision on dedicated latent process positions, closer to LOTUS. If process supervision succeeds, a later phase can add a separately refined answer state, closer to Tiny Recursive Models, while preserving the compressed-memory substrate.
