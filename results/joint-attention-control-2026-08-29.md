# Joint-context attention control — 2026-08-29

## Provenance

- Model: `HuggingFaceTB/SmolLM2-135M`
- GitHub Actions run: `33231710117`
- Head: `6320e9f6c3a603e8cc8eacf75a2fc0da791fd5cc`
- Artifact: `9708717039`
- Artifact digest: `sha256:07b58ceff33b2cf7514682cfa7f14544b87e4ee02470e8ab70a4fdecb91edaab`
- Cases: 120
- Candidate memories: 6
- Design: all candidate memories are concatenated into one shared context; query-token attention is measured to each candidate's token span.

## Results

| Method | Top-1 | Top-3 | MRR | Median rank | Selected-answer NLL |
|---|---:|---:|---:|---:|---:|
| gzip conditional | 0.6333 | 0.9250 | 0.7800 | 1 | 6.5288 |
| **SmolLM2 conditional description length** | **0.8750** | **1.0000** | **0.9361** | **1** | **5.2170** |
| joint attention: last-layer mean-head mass | 0.1833 | 0.4500 | 0.4042 | 4 | 8.2488 |
| joint attention: all-layer mean mass | 0.2667 | 0.7500 | 0.5422 | 2 | 7.9693 |
| joint attention: last-layer max-head mass | 0.1417 | 0.4917 | 0.3889 | 4 | 8.2385 |

On paired Top-1 outcomes, CDL versus gzip has 33 cases where CDL alone is correct and 4 where gzip alone is correct (exact McNemar p ≈ 1.08e-6). CDL versus all-layer mean attention has 73 CDL-only wins and 0 attention-only wins on this benchmark.

## Interpretation boundary

This result supports the claim that **raw attention weight mass is not a strong explicit relevance score for this benchmark**. It does not establish that CDL is a replacement for Transformer attention. Attention is an internal computation mechanism trained end-to-end; individual or averaged attention weights are not guaranteed to correspond to human-labelled relevance.

The architecture hypothesis therefore remains narrower: conditional description length may provide a better *training target for routing* than direct relevance labels or raw attention-weight heuristics.

## Failure note

CDL still fails on 15/120 cases. The failures concentrate around same-entity wrong-relation distractors, especially queries involving `head of government`, where linguistic overlap with capital/government facts can make the wrong relation compress the query well. This is useful negative evidence and motivates natural-language/paraphrase tests.
