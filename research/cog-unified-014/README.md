# COG-UNIFIED-014: verified research evidence

**Developmental result; no architecture promotion or official campaign credit.** 42 new training fits, 15,450 optimizer updates, 458,400 training histories, 6,412,800 event exposures. All 42 checkpoint reload evaluations reproduced their recorded results.

## Downloadable evidence

- [Source, protocols, aggregate results, closure audit, verification and checkpoint manifest](archive/014_source_and_results.zip)
- [Initial custody: 12 checkpoints](archive/014_initial_custody.zip)
- [GRU controls: seed 173](archive/014_gru_173.zip) and [seeds 271/907](archive/014_gru_remaining.zip)
- [Curriculum and lexical matcher: 12 checkpoints](archive/014_curriculum.zip)
- [Presence-coordinate repair: six checkpoints](archive/014_presence.zip)
- Integration: [seed 173](archive/014_integration_173.zip), [seed 271](archive/014_integration_271.zip), [seed 907](archive/014_integration_907.zip), [three protected-only controls](archive/014_integration_protected.zip)
- Historical [Gate 0C custody source, results and all 18 actual checkpoint binaries](archive/gate0c_custody_and_source.zip).

All listed ZIP bundles have been uploaded as Git binary objects and retrieved, with full base64 content equality against the original local bytes. Extract all 014 bundles to a common directory, preserving paths, then run the included `verify.py`. See the included README and checkpoint manifest for exact instructions.

## Findings

Initial final-map-only event parsing failed. Factorized lexical matching plus a separate presence coordinate and continuous value proposal enables exact hard custody on the tested one-entity sentence templates, from final-map supervision alone. Six presence-based checkpoints pass all tested fresh panels; all 23,040 supported action and address decisions and 7,680 write cases in the exhaustive presence-domain audit are correct. All six fail the two-entity stress. No unrestricted language guarantee. Hybrid COG and protected-only controls solve the controlled retrieval task using the same frozen supervised parser; continuous-only and memory ablation fail, so no added COG benefit is established.

## Historical archive gaps

The historical audit deduplicates 159 prior checkpoint hashes across Gate 0, 0B and 0C. 123 are locally recovered across the mounted archives. Only the 18 Gate 0C custody checkpoints have been remotely byte-verified in this publication; 105 recovered originals remain pending remote publication, and 36 are missing from the available mounted archives. Earlier cycles 001–012 were not recovered here. Do not count the duplicate proof-of-transfer checkpoint as a separate fit.

## Prospective custody policy

Each new experiment must preserve source, configuration, actual weight bytes, raw evaluations and hashes, then pass checkpoint reload and remote retrieval verification before campaign admission. Record negative outcomes and distinguish training from replays. The source archive contains a proposed ARCHIVE_POLICY.md and a non-clobbering authenticated `archive_release.py` utility for large assets. This utility was not executed here because the environment has no authenticated GitHub CLI or direct release-asset upload access. The branch does not change the canonical architecture or automatically update lab automations.
