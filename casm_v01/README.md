# CASM v0.1 — Compression-Attention Small Model

CASM is a causal byte-level LM trained entirely from scratch to test whether **conditional description length can improve information routing** inside a small general-purpose sequence model.

v0.1 corrects two failures exposed by the 600-step v0 ablation:

1. The objective now upweights answer bytes, so success requires solving tasks rather than merely compressing prompt templates.
2. The learned predictive-code teacher uses a shared base predictor plus a memory-conditioned residual, making base-vs-conditioned code lengths counterfactually comparable.

Additional controls:

- the post-routing FFN is active in all ablations, removing the v0 extra-depth confound;
- memory residual gates start more conservatively;
- gzip-to-learned-teacher handoff is confidence-adaptive;
- training ramps to a 50% hard-example curriculum;
- training uses eight chunks and evaluation uses six-chunk streams;
- evaluation reports answer NLL explicitly.

The paired ablations remain: local-only, Q/K compressed memory, and compression-attention memory.
