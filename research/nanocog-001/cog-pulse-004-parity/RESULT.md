# COG-PULSE-004: exact-source repair parity

Both draft repairs passed one bounded CPU engineering check on the exact original and repaired source. The run used freshly initialized parameters and constructed inputs, not trained checkpoints or held-out data.

| Check | Result |
|---|---|
| Publisher: 32-token branch rollout, 17 possible actions, three-token prefix | **1,074 → 544 actual vocabulary predictions** in each of six mode/seed cases. Of the 530 omitted calls, 527 were discarded branch predictions and three were discarded prefix predictions. |
| Publisher: one-token horizon, no prefix | **17 → 17** in each of six cases. |
| What changed in the Publisher outputs or backward pass? | **Nothing in these fixtures.** Requested logits, actions, intermediate state, two losses and named gradients matched bit for bit; unused-gradient patterns and the CPU RNG state were unchanged. |
| Incremental graph update vs full recomputation | **165/297 mismatches** with the original code, **0/297** with repaired code. Across seven graph modes, three seeds and the supported dense/grouped inference paths, the largest difference was **2.02×10⁻⁶** in intermediate states and **7.75×10⁻⁷** in output logits. |

The graph repair caught edge and operator edits even when no leaf value changed. A rewire to a different equal-valued source still triggered the right invalidation. It did **not** make the cache self-verifying: supplying the wrong previous inputs alongside cached state could still return a stale answer. The original graph is fixed-layered and receives dependencies; this study did not test learned dependency discovery or arbitrary graph cycles.

This establishes preservation for tested source-level behavior and removal of redundant predictor calls, not measured latency, trained task quality, stable learned identity, or an attention replacement. The next graph gate is a cache receipt that binds prior inputs, cached state, model and execution settings before reuse. The next learning gate is a separately admitted comparison that tests whether the publication objective learns stable identities rather than only fitting supplied routes.

Provenance: [source-repair checkpoint](https://github.com/FPC-effortless/veritas/issues/482#issuecomment-7118962720). Exact engineering protocol ledger SHA-256 `acb1529f18b9c85ab8bdaacb9df90ac4ec328c96c244f655ccf6df6a9b20cf42`; native result SHA-256 `9a499e49a73e743b8398d1d0f36bffe52690a37d041c62e89b1a0b7afc67b589`. No checkpoint, protected test, optimizer step, or paid compute was used.
