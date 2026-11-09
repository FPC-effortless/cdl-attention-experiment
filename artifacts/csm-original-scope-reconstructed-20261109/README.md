# CSM original-scope archive: reconstructed delivery

This package publishes the verified original scientific work: 30 frozen checkpoints and 242 raw evidence files, source history, results, mathematical model, Lean project and parallel-processing analysis. It excludes CSM-ADDRESS-002 and its 12 later checkpoints.

**This is not commit 168d44d and not a byte-identical reconstruction of its final packaging.** That consolidation commit, final index and packaging bytes were unavailable. The original scientific archives are preserved unchanged and verified. This new index and copied recovery utility are explicitly reconstruction tooling, not recovered original consolidation files.

## Contents
- `CSM-FOUNDATION-001-review.zip`: original source Git bundle, readable patch, report, audits, all 30 checkpoints and 242 raw evidence files.
- `CSM-GRAPH-001-verified-project.zip`: original mathematical report, formal source, verification records, runtime/parallelism analysis, source bundle and snapshot. Its six checkpoints duplicate a subset of Foundation, not six additional fits.
- `MANIFEST.json`: archive hashes, provenance, counts and packaging caveat.
- `restore.py`: verified recovery utility copied from the later recovery package; no later experiment data included. Restores 422 original source/evidence paths and refuses differing-file overwrites.

## Restore
First run `python3 assemble.py` to join the two Foundation archive parts. They preserve the original ZIP exactly; splitting is solely for upload size limits. Then from this directory run `python3 restore.py --target /absolute/path/to/empty/recovery`. Use `--verify-only` first to verify archives and check destination conflicts. Git bundles preserve exact original source commits; restore mapping also includes original raw evidence omitted from tracked Git files.

Exact source commits: Foundation `a89ae4e516627b1b5030374df97a2a9a9b6f19c6`; Graph `0e667ffd17f6904097a38be37455c2c4350b65a0`, its direct child with 31 formal-project additions. Bundles require base `c26ff92b8ecec74045bd2393cdaf956c99a0d9d5` from this repository.

## Findings and limits
The original CSM failed its structured learning screens and the tested parallel implementation was slower. Successful integrity checks and conditional Lean proofs do not establish learned competence or an inference advantage. No new training, scientific selection, merge or deployment is performed by this archival publication.

## Verification
All ZIP CRC checks, Foundation 261 entry hashes, Graph 192 snapshot hashes, 242 raw evidence records and 30 checkpoint counts were checked. 180 tracked snapshot files match Graph source exactly; 12 additional snapshot files match original Foundation evidence. The manifest is not a claim to reproduce unavailable commit 168d44d.
