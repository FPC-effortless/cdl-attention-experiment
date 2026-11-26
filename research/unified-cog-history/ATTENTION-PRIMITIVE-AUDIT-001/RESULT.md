# ATTENTION-PRIMITIVE-AUDIT-001 — fair primitive comparison (2026-11-26)

This developmental audit uses the actual recovered UNIFIED-017 reference COG block (pinned source commit 671a653ab37752531176043fbeae39270562c7ed, reference/lm.py), CTG disabled, division-free associative recurrence. One layer, width 32; train on eight distinct random keys from 16, independent values from eight classes, four queries appended to the history, no answer feedback. Complete key/value record prepacked into one vector. Shared input information and learned key/value/type embeddings. Answer cross entropy only, AdamW .003, clip 1, batch 24, three seeds 6101/6121/6131, 500 updates. Match active parameters by adjusting FFN width. Primary gate: every four-head SDPA seed >=95% individual and >=90% complete-history accuracy on 512 fresh eight-record histories. Both attention baselines passed in all seeds.

| Architecture | Params | Updates | Individual correct | Complete histories | Relevant mutation pairs | Nuisance mutation pairs |
|---|---:|---:|---:|---:|---:|---:|
| COG | 12009 | 500 | 3560/6144 | 125/1536 | 52/1536 | 85/1536 |
| SDPA one head | 11979 | 500 | 6144/6144 | 1536/1536 | 1536/1536 | 1536/1536 |
| SDPA four heads | 11979 | 500 | 6144/6144 | 1536/1536 | 1536/1536 | 1536/1536 |
| COG continuation | 12009 | 2000 | 3849/6144 | 213/1536 | 137/1536 | 169/1536 |

The continuation was added after primary results, resumes the same optimizer and RNG, and is not an independent fit. Attention already achieves perfect accuracy at 500 steps. COG improves on average with extra training but does not close the gap. This establishes a developmental retrieval-learning deficit, not a proof of incapacity at larger scale or under another optimizer. A single attention head suffices for sequential queries; simultaneous independent retrieval is not isolated here.

| Records | COG 500 individual accuracy | COG 2000 | SDPA one head and four heads (each) |
|---:|---:|---:|---:|
|2|89.71%|90.72%|100%|
|4|76.40%|79.10%|100%|
|8|57.94%|62.65%|100%|
|12|46.42%|50.63%|100%|
|16|40.56%|43.18%|100%|

Note: record-count panels outside eight are OOD; two-record panel uses two queries. Fixed COG carried state = 256 bytes FP32 per sequence for one layer; conventional full KV cache at width 32 adds 256 bytes per retained token. Memory budgets differ, training FLOPs not matched. Causality checks pass. Primary COG split maximum deviation 2.3842e-6; zero prediction disagreements. All 12 saved checkpoint evaluations and hashes reproduce. Nine distinct fits + three continuations; 9000 optimizer updates total, 216000 sampled histories, no campaign credit assigned. No GPU, paid compute, language modeling, protected test data, full architecture promotion, or universal generalization claims.

Scientific decision: retain COG's existing bounded state and nonlinear computation; prioritize basic query-conditioned historical access under interference, competent matched baselines, and state-capacity-matched comparisons. Do not use arbitrary semantic-governance, long-context or reasoning tasks as rejection gates unless competent controls solve them under matched conditions. UNIFIED-017 neural-only 1024-event held panel was 0/48 for both COG and SDPA; nondiscriminating at that budget.

Local full evidence archive: cog_attention_primitive_audit.zip, 2331636 bytes, SHA256 f705a6331ce1219ef2f702c5c8a12009e3575b2a391a0477c8c2292bd3e23da1. Contains source, protocols, raw results, original optimizer-bearing checkpoints and lossless inference exports. Remote status: this result record only; exact new checkpoint/source archive upload not verified in this cycle. Prior historical publications unchanged.
