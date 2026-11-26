# Unified COG research evidence and checkpoint custody

This branch is a developmental research record for the conversation's UNIFIED-001–016 series, including the separate CTG/temperature screening cycle. It is not an architecture promotion and does not modify the canonical main branch or the official campaign ledger.

## Required custody for future COG fits

1. Freeze protocol, source revision, environment, data fingerprints, configuration, seed and matched controls before launch. Keep copyrighted, private and protected test contents out of public releases.
2. Retain the actual trained checkpoint bytes, evaluation results, reproducibility instructions and SHA-256 manifest. Save optimizer state if continuation is intended. Do not overwrite an existing fit ID.
3. Publish source and sanitized results on this evidence branch or an owner-approved research branch. Publish large checkpoint bundles as versioned GitHub release assets, with explicit checksums. Git LFS is an alternative only if authorized and supported.
4. Re-fetch the asset or byte digest from GitHub and compare with the local SHA-256; record its immutable commit/release URL. A manifest or local attachment alone is not remote custody.
5. If transfer is blocked, mark LOCAL_ONLY / UPLOAD_BLOCKED; do not report weights as published, count replays as fits, or promote an unqualified mechanism. Preserve negative and superseded evidence.
6. Keep verified campaign counts separate from historical checkpoint members, intermediate snapshots, inherited weights and no-fit diagnostics. Respect Doctrine V2.1 and matched-quality lifecycle measurements.

## Current status — 2026-11-26

Local originals from cycles 001–015 and the separate temporal 010 screen: 21 ZIP archives, 21 CRC checks passed, 755 checkpoint members, 732 distinct checkpoint SHA-256s. Cycle 001 had no retained trained weights. Historical B004/B005 checkpoint custody is a separate unresolved problem; this inventory is not the Council's full lab ledger. Remote binary custody is not yet verified. UNIFIED-016: six new ~1M language fits and fifteen synthetic semantic custody fits, all 21 checkpoint evaluation replays passed; no architecture promotion.

Historical archive index and selected results are separate files in this folder. GitHub text records are not substitutes for actual release assets.
