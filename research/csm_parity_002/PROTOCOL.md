# CSM-PARITY-002: forward parity and exploratory learning pilot

Starting source: b5ff1410905dc5dfb11a383ebb60362247899070.
Primary stage: reproduce NumPy checks and compare archived hard forward with
mapped identical arrays. Regulated adapter preserves archived operations and
straight-through estimator; its hook is a correctness adapter, not an optimized kernel.

Exploratory pilot registered before fitting: authentic archived Tiny Shakespeare,
SHA256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Use original train [0,1003854), exposed dev [1003854,1059624); never score test.
Byte vocabulary 256. Three matched arms instant/single/multi, seeds 11/23/37,
32 AdamW updates, lr .001, weight decay .01, batch 4, chunk 64; contiguous
2049-byte train streams selected per seed. Carry all state across chunks,
detach between updates (truncated gradients). No document reset within streams.
Same base/driver/injection initialization and samples per seed. Nonzero random
injection initialization. All arms same parameter count and nominal operations;
record measured CPU time, not a claim of equal actual FLOPs.
Final weights only; no tuning or checkpoint selection. Evaluate first 1025 dev
bytes as one stream, report next-byte NLL plus trace reset and cross-stream swap
interventions (swap requires multiple independent streams; deferred here).
Reset-only intervention clears trace before each dev token, preserving records.
No original smaller CSM superiority claim. A small pilot on exposed dev is not
scientific qualification. Save full model/optimizer/RNG checkpoint and replay
NLL. No paid compute, GPU, protected test, merge or deployment.
