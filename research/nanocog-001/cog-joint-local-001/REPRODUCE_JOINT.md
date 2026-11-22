# Replay COG-JOINT-LOCAL-001

Use Python 3.12.14, PyTorch 2.10.0+cpu, and the pinned Veritas
TrainLab source from commit `eec439492407b0aa26922c23b5ae40d95289f4eb`.
Set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, and use
the corresponding exact runtime hashes in the fit audit. The private ZIP
provides `SHA256SUMS`; verify it before reading recipes. The published code is
not sufficient on its own to recover the private inputs or saved weights.

To reproduce the aggregation with no model calls, work in a clean copy of the
archive and move the existing score result to a different name. Retain the seal,
original actor receipts and original file/directory names. Run:
```sh
python score_joint_local_001.py \
  2e1d5dcfa332f5c11536f9fbe1b62822dce3e9955aface958b8bb055261d89ff \
  6d47358658fa9037427427a763ef58a90909d386e13c7197849b193eadc2016c \
  9b900c2b93c359161de53ead5a80373a07390e1da3ab4176ce250a1b45624d53
```
Expected result SHA-256:
`ef20b2a45b95038ae22edc7e4cd313df9f5ff1e1a81ec16771cfebd45d423293`.

For native replay, use each exact named `recipe.json` under the TRAIN or
actor admissions directories with `python -m trainlab run RECIPE --runs
FRESH_RUN_STORE`. Keep truths out of actor closures and holdouts out of
TRAIN. Receipts may differ in timestamps and run IDs; report mismatches in
tensor or logit hashes instead of picking a different run. The bound score
is for the retained sealed outputs, not automatically authorized for a
different replay. This cohort has now been used and is not a fresh test set.
