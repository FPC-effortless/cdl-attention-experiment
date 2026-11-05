# LRR-003B — Stateful incremental retrieval cost protocol

Pre-registered before dedicated CI measurement. Based on the frozen LRR-003 code and **not** allowed to change the trained model or infer gold data from answers at inference.

Question: Does an exactly equivalent causal suffix computation allow existing learned addresses to be built once and reused across repeated questions, instead of re-encoding the whole context every time?

- Encoder: causal 3-layer dilated convolution with 15-token receptive field. Retain exactly 14 trailing input tokens, append new observations with only that left context, and write hard inferred key/position postings.
- Query: encode only the 14-token history suffix plus the question, retrieve one address, return absolute token indices. Query must not mutate the history state. No question or gold answer field enters the index writes.
- Tokenization and punctuation are unchanged from LRR-003. A separate query() API call is an explicit **question boundary prior** and is reported as such.
- Mechanical gate: chunked and whole-history appends must produce identical postings on randomized untrained model controls, and the query address must match full recomputation on the same text.
- Fixed profiled model: train exactly the frozen LRR-003 seed21 with 700 optimizer steps and batch48, no tuning. H in {6,32,128,512}, one static raw history and the same query repeated 24 times per H. The question itself is unseen train alias-pair test (2,3).
- Report CPU model parameter count and training duration; cold prefill ms; cached query median/P95 including its fixed receptive context; naive full-route median/P95 including full re-encode/rebuild; correctness/mismatch; amortized 1 and 64 query wall-cost estimates.
- Complexity target: ingest O(N) each new text, storage O(N), repeated query O(|Q|+R) encoder and expected O(1) hash + O(k) result, where R=15 and Q is query length. This is *not* a proof of O(1) ingestion/total computation.
- No inference performance claim from hash probes alone; wall-clock and preprocessing costs must be displayed. No GPU kernel/LM throughput or attention-parity claim.

Because LRR-003 showed seed instability, seed21 is a mechanistic cache witness only. The caching intervention does not repair semantic quality or abstention failures.