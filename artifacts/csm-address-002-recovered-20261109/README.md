# CSM-ADDRESS-002: recovered research publication

This package preserves the additional CSM-ADDRESS-002 experiment and the
unchanged original research, recovered from the saved 9 November 2026 export.

Start with the [ADDRESS-002 results report](CSM-ADDRESS-002-results.md).
Explicit training-only controller supervision enabled the unchanged CSM to
solve the fixed SET/recall panels in three seeds; answer-only training remained
weak. This does not establish language competence, general composition,
architectural superiority, or an inference-speed advantage.

## Contents and provenance

The four archive parts reassemble into the exact original
`CSM-Research-Update-20261109.zip`: 24,409,036 bytes, SHA-256
`e4a3d3e39e6e8aa3e6f4f3c6c76b705dacf310be4adb8fc82f068712e678b9f2`.

Its 305 files include:

- 148 ADDRESS-002 files: all 12 new trained checkpoints, optimizer/RNG states,
  frozen synthetic datasets, raw predictions, logs, implementation, protocol,
  results, reviews, semantic contract and preserved evidence-recovery incident.
- Unchanged foundation and formal source dependencies.
- Both unchanged original archives, containing the 30 original checkpoints and
  242 original raw evidence files. The combined record has 42 checkpoints.
- Original restoration instructions, manifests and research index.

The source archive records commit
`1e658d61bed7e629be99524d87a6c7a1bdc8b358` and preregistration
`5bf17652ae478e7975217a30f85cc7d418eec3dc`. This is a new publication of the
verified file export, not recovery of those exact Git objects or of unavailable
consolidation commit `168d44d`.

The branch builds on the previously published original-scope package at
`1bb6e10084cb06a9b600c2d055bfd54a9ae4ee0d`, without changing any base files.
The [publication manifest](MANIFEST.json) records part hashes and scope; the
[unchanged source manifest](SOURCE_ARCHIVE_MANIFEST.json) records source hashes.
The standalone results report is byte-identical to its archived copy.

**Historical status text is preserved.** Statements that publication was blocked
in the archived report, index, start document and source manifest describe the
earlier source snapshot. This README describes the recovered package on this
branch. Frozen scientific files were not edited to update publication wording.
No merge, deployment, new training or new scientific claim is included.

## Assemble and restore

Download or clone this branch, then run:

```bash
python3 artifacts/csm-address-002-recovered-20261109/assemble.py
```

The helper verifies every part and the assembled archive, and refuses to
overwrite a different existing archive. Extract the resulting ZIP into a new
empty directory. ADDRESS-002 and its model dependencies are already expanded.
To restore the original frozen checkpoints, run from that extracted root:

```bash
python3 artifacts/csm-archive-20261109/restore.py
python3 artifacts/csm-archive-20261109/restore.py --verify-only
```

## Checks performed for publication

- Source ZIP integrity passed. All 303 source-manifest entries matched their
  recorded byte counts and hashes; all 305 exported files matched the ZIP.
- Both nested original archives matched the previously published archive hashes.
- Independent packaging review verified restoration of 30 original checkpoints
  and 242 evidence files, 261 foundation checksums and 192 graph snapshot hashes.
- All 12 new checkpoints were present; five focused data-integrity tests passed.
- All 26 recovered Python source files passed syntax parsing.
- No obvious credentials were found in the reviewed text. This is not an
  exhaustive secret certification.

The recorded 39,936-prediction replay is preserved, not rerun for publication.
PyTorch is unavailable in this publication environment. Lean was not recompiled.
No full model replay, fresh training or runtime benchmark is claimed here.

For replay, use the recorded Python 3.12.14, PyTorch 2.5.1+cpu and NumPy 2.3.5
environment on a separate disposable extraction:

```bash
cd experiments/csm_address_002
python3 -m unittest tests_data verification -v
python3 postfit_audit.py
```

The audit writes review receipts, so keep the published frozen files unchanged.
The completed training runner refuses training after closure; preserve that
boundary and all original attempts.
