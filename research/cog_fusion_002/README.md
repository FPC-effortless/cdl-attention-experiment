# COG Fusion 002 — learning across generated answers

This study tests a specific failure found in COG-FUSION-001: later observations
could be interpreted incorrectly because the backbone carried state from its
own generated answers, while training histories had ended at the first answer.

**Status: incomplete campaign.** Five fits reached update 1,200. The original
continuous seed 17 terminated with signal 9 after 13 recorded updates. Its one
permitted replacement lost its execution handle after 107 recorded updates,
without a completion or exit receipt. Its terminal status and cause are unknown.
No held model performance has been scored, and no qualification decision exists.

Read `RESULT.md`, `evidence/INCOMPLETE_CAMPAIGN_AUDIT.json` and
`review/INTERRUPTION_DISPOSITION_REVIEW.md` for the retained evidence and the
conditional, unexecuted continuation proposal.
The implementation, model size, inherited tokenizer and memory executor are
unchanged. The experiment changes the training histories and compares two
matched continuation policies. Neither an unsuccessful finite fit nor success
on these controlled histories establishes a universal architecture result.

## Comparison

Each seed starts both arms from its corresponding 001 memory endpoint. Both
receive the same 2,048-family TRAIN corpus, 1,200 updates, supervision and
language rehearsal. `continuous` carries the actual generated answer state;
`reset_training` discards all neural state after each training answer while
retaining the factual ledger. Both arms are evaluated continuously on held data.

The correct-answer loss uses a separate teacher-forced branch. Its state is
discarded, so labels cannot secretly supply the continuation seen by later
observations. Every actually emitted character is consumed once, including the period when
emitted; generation retains the twelve-character cap and does not force a terminator. The first 100 updates use supervised typed execution; the remaining 1,100
use predicted writes and queries. All evaluations use predicted execution.

The full specification and gates are in `PROTOCOL.md`. Training adequacy,
familiar repair, specific causal advantage, novel wording, earlier regressions,
paired-parent language preservation and original-full recovery are separate
decisions. The old held panels are explicitly exposed regression evidence.

## Restore

Checkout the source commit identified in the evidence archive, then extract the
archive at the repository root. This makes both sibling studies' required
evidence files available without duplicating their Git-backed implementations.

```bash
cd research/cog_fusion_002
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../cog_fusion_001/requirements.txt
```

The recorded runtime is Python 3.12, Torch 2.10.0+cpu, NumPy 2.3.5 and pytest
8.4.2. Matplotlib is required only to regenerate plots. All fitting and serving
use CPU. The model has 997,674 active parameters, 1,077,834 stored parameters
and 10,592 neural-state bytes per world. Durable records and receipts can grow.

## Use a saved endpoint

From this directory:

```bash
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed29/final.pt --db demo.sqlite observe "Amara owns the chest."
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed29/final.pt --db demo.sqlite observe "The coin is in the chest."
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed29/final.pt --db demo.sqlite ask "Who owns the container holding the coin?"
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed29/final.pt --db demo.sqlite observe "The chest belongs to Bela."
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed29/final.pt --db demo.sqlite ask "Who owns the container holding the coin?"
```

Seed 29 is used because its continuous endpoint completed; no held scores were
used to choose this example. It is an unqualified research checkpoint. Every
command starts a fresh process and resumes the saved session. Use a new database when
changing checkpoints; interpretation identities must agree with saved state.
The supported entity vocabulary is unchanged from 001 and inputs use explicit
observe/ask boundaries. Consult the measured failure cases before treating this
as a general-purpose agent.

## Reproduce engineering and retained-fit checks

```bash
python -m pytest tests -q -ra
python review/audit_incomplete_campaign.py --output fresh_incomplete_audit.json --snapshot-dir fresh_incomplete_snapshots
```

The separate incomplete-campaign audit performs no inference and scores no
TRAIN probe or held data. It checks the five complete final/optimizer states,
replays their draws and accounting, and records the two incomplete attempts.
The unresolved canonical files stay in place; a stable byte snapshot records
what was observed without asserting process termination.

The original `review/audit_training.py`, `run_evaluations.py` and `summarize.py`
retain their complete-campaign requirements. They cannot admit this inventory.
Do not fabricate `RECOVERY_RESULT.json`, remove the recovery marker, or treat the
five available endpoints as the planned three-seed comparison. The complete
model evaluation, language preservation, persistent-session and intervention
scripts remain available as source, but their planned outputs were not produced.

A future continuation requires reliable execution-status evidence and a separately
recorded bounded amendment. The disposition review specifies those conditions;
no additional attempt or amended evaluation has been executed.

## Fit command for a future, separately recorded experiment

```bash
python train.py --parent ../cog_fusion_001/evidence/runs/memory_seed17/final.pt --train-data evidence/data/train.json --shakespeare ../cog_fusion_001/evidence/reproduction_inputs/train.bin --output new_continuous_seed17 --arm continuous --seed 17 --steps 1200 --teacher-steps 100 --pairs 4 --threads 1 --cpu-cap 3600 --native-lr .000075 --lr .0005
```

This command is provided for reproducibility and has not been used to start an
additional replacement. Resolve the execution-status problem and record any
new attempt policy first. Use `--arm reset_training` and another empty output
directory for a new matched control. Preserve the source and data hashes. An extended or otherwise modified
run is a new experiment rather than a replacement for the locked endpoint.
`run_campaign.py` refuses an existing training lock or existing output directories.

## Evidence map

| Evidence | Location |
|---|---|
| Fixed protocol, model/data/regime admission | `PROTOCOL.md`, `evidence/PRETRAIN_REVIEW.json` |
| Immutable source and configuration lock | `evidence/TRAINING_LOCK.json` |
| Initial tensor parity | `evidence/PAIRED_INITIALIZATION.json` |
| Data, isolated TRAIN file and labels | `evidence/data/` |
| Five complete endpoints and the unresolved replacement trace | `evidence/runs/` |
| Individual completed-fit integrity and incomplete work | `evidence/INCOMPLETE_CAMPAIGN_AUDIT.json` |
| Observed recovery snapshots | `evidence/incomplete_snapshots/` |
| Planned generated multi-turn evaluation | Not run |
| Planned old familiar/fresh/long regressions | Not run |
| Planned ordinary-language measurements | Not run |
| Planned adapted-checkpoint persistent sessions | Not run |
| Planned preselected state interventions | Not run |
| Planned qualification and family bootstrap | Not run |
| Independent protocol, data and arithmetic reviews | `review/` |

The original interrupted attempt is retained under `evidence/failed_attempts/`.
The once-only recovery began with the identical locked fit, parent and seed,
and remains unresolved. Its marker is preserved. No second replacement, held
selection, budget extension or fabricated exit receipt is admitted. The report
accounts for both incomplete traces as observed compute lower bounds.

The language arithmetic note preserves the locked scorer's raw flags while the
independent summary applies the protocol's inclusive percentage gates directly
to the saved decimal NLL values. Any disagreement is reported explicitly.

## Scope

This cycle uses fixed entities and controlled templates. The shifted-schedule
session is a diagnostic, and the LONG panel inserts neutral events. These tests
do not establish arbitrary reasoning length, new-entity vocabulary, an attention
efficiency advantage, or joint language-driven numerical computation. The
optional 001 numerical wrapper is unchanged and is outside this repair fit.
