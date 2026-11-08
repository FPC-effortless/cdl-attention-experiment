# Recovered original LRC route repair

This prerelease preserves the original frozen repair, checkpoints, dependency
archive, audit evidence, and restore instructions. It is an artifact publication;
no training or new refinement experiment was run to produce it.

## Original references

- Repair branch: `research/lrc-route-repair-002-20261108`
- Repair HEAD: `3f71af6eb53827eb7eb3c74e10964f70e8e84460`
- Frozen source commit: `6c3976efc2382e28731009f9637ecf31872e5c13`
- Portable handoff branch: `research/lrc-route-repair-002-handoff-20261108`
- This release's target: handoff commit `566ba9933a35cf88a55f830891d8d6e33d6ca524`

The handoff contains the repaired source and all downloadable assets together.
The repair branch preserves the complete original research history.

## Download and restore

Download the two ZIPs, the code patch, `SHA256SUMS`, `restore.py`, and
`requirements-restore.txt` into one directory. Read the attached `README.md` for
the clone and environment setup. `python3 restore.py --verify-only` verifies the
original bytes; `python3 restore.py` extracts the archives without training.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| LRC_ROUTE_REPAIR_002_Checkpoints_and_Evidence_20261108.zip | 5,489,762 | `06ad6c3d1e6647fa4912ec8631ca7367e5e4c775420f3573c1a024ea5f50daf6` |
| LRC_ROUTE_REPAIR_002_Code.patch | 184,354 | `1b25a4582994ec947fc7b2407470efe546640c744747d7a34a32f3e2d18a71eb` |
| lrc_vb001_matched_comparison.zip | 7,358,645 | `20196fc1c54a7fcfc177d977883728fe5e07edccb6491a8b8e8921c5e4efe3f5` |

The evidence includes twelve confirmation checkpoints (four conditions, seeds
901–903) plus separately labeled development checkpoints. Historical comparison
results are not part of the repaired cohort. Publication does not establish a
new capability or performance result and does not mark the repair as merged.
