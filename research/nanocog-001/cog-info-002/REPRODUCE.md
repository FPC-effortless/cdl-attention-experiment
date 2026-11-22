# INFO-002 replay

Scope: frozen local intervention on the nanoCOG gated32 ROLE-002 model,
not a fit or the held MAIN gate. Start in a fresh directory with the
reviewed actor recipes, inputs and frozen weights in `info002_admissions`.
Use Python 3.12.14, PyTorch 2.10.0+cpu and the native TrainLab runner from
Veritas commit `eec439492407b0aa26922c23b5ae40d95289f4eb`. Set OpenMP, MKL and
OpenBLAS thread counts to one. Preserve recorded run directories read-only.

Verify the admission ledger SHA-256:
`425737b22fb1916dacedaba9aa630c779cb9be193e03844849ab352f4f208304`.
Each recipe lists exact source and input files; their individual hashes
are retained in the ledger. Run the 1171 and 1371 recipes separately with
`python -m trainlab run info002_admissions/info002-SEED/recipe.json --runs info002_runs`.
Save each native command's JSON as `cli-output.json` in the matching
admission folder. Replay from only the exact admitted inputs. Do not mix
the separate evaluator panel into the actor's available files.

Compare replay artifact hashes and control counts against the sealed
records. Run IDs and timing receipts need not match across machines.
Report divergences instead of selecting a rerun. Never overwrite the
original native outputs. The packaged score should be exactly reproducible
from sealed, already generated logits without installing Torch or loading
the checkpoints.
