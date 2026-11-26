# COG-UNIFIED-020 — developmental query decomposition
Date: 2026-11-26. Parent: UNIFIED-019 commit 04cfd10f81b98eeb4f39782acbb5d2aacfc358da (priority archive). Recovered active FactorMatrix/FactorAttn methods, constructors, and scan from the parent archive. Fresh initialization; no exact previous-weight replay and no canonical promotion.

## Accounting
33 independent training fits, 29,700 optimizer updates, 950,400 training-history exposures, 33/33 exact evaluation-count replays. CPU-only. Six separate reduced-grid numerical diagnostics; full kernel grid timed out. No official campaign credit assigned.

## Query decomposition results
Each arm has three seeds, 900 updates, batch 32, ~12K parameters (MLP 12,981). Held-out ordered pairs satisfy (q0+3*q1)%5==0 and are excluded from training. Each reported panel has 768 histories. The common query vector is learned from two ordered key embeddings; only factorized references get separate key embeddings at the read interface. COG state 640 B, narrow attention prefix cache 640 B at 8 records, wide attention 2,048 B. Counts exclude temporary activation/current query.

| Initial arm | Both values, 8 held-out | Aggregate | Both values, 16 held-out |
|---|---:|---:|---:|
| Factorized attention | 100% | 100% | 100% |
| Factorized delta | 100% | 100% | 100% |
| Unified narrow attention | 100% | 100% | 100% |
| Unified wide attention | 100% | 100% | 100% |
| Unified linear delta | 59.64% | 46.35% | 40.76% |
| Unified MLP delta | 15.23% | 28.91% | 11.07% |

Fresh-seed repair phase: direct address heads, separate banks, and contributor-aligned geometry.

| Repair arm | Both values, 8 held-out | Aggregate | Both values, 16 held-out |
|---|---:|---:|---:|
| Factorized delta | 100% | 100% | 100% |
| Unified narrow attention | 100% | 100% | 100% |
| Direct shared delta | 19.40% | 69.40% | 15.23% |
| Direct split banks | 19.01% | 77.99% | 14.97% |
| Direct with contributor CE (weight .5) | 98.96% | 93.62% | 81.38% |

Aligned arm held-out complete counts 255, 252, 253 of 256; seen complete counts 251, 250, 250 (misses 98% threshold in two seeds). Independent intervention control 750/768; same head 88/768; swapped 81/768; uniform 25/768; zero message 21/768. Contributor-supervised repair passes held-out near-attention gate only, not autonomous acquisition or full gate. No promotion.

## Execution
Blockwise rank-one delta recurrence uses unit lower triangular system (I+tril(beta Gram,-1)) delta=beta(V-KS0); cumulative rank-one states. Six FP32 CPU tests, block 16, widths 16/32/64, lengths 128/512, batches 1/4. Max absolute error vs FP64 recurrence 4.663e-7, gradient 7.451e-9, segmented continuation 7.153e-7. Forward-only speedup 4.57–7.60x against Python loop, not optimized fused attention or end-to-end. Full grid timed out.

## Evidence custody
Local verified archive digests (these are NOT claims of GitHub binary custody):
- cog_unified_020_evidence.zip (4,897,957 bytes): SHA256 61a85c202fd2b5a2644dbe6c0ccc91d6c2f3205c17652bf538854e541c07aadd
- cog_unified_020_inference_release.zip (1,530,194 bytes; all 33 lossless Safetensors): SHA256 5e6299ae4896ebff4e267c261d6b30f2041511f31637ad09f30d42bfba590e19
- cog_unified_020_priority_models.zip (427,836 bytes; nine repair controls and aligned candidates): SHA256 dca215154a224d87bc43cf46b22f379b7e4da79aebaf50bf044d39e6c9144126

## Decision
Retain factorized delta as the structured reference. Learned autonomous query addressing remains unresolved. No new unstructured induction or LM integration training, no GPU, no FLOP-matched qualification, no canonical integration. Historical binary backfill remains open. Register this as partial developmental evidence, not official campaign credit or completed UNIFIED-020 qualification.