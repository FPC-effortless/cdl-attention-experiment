# Reproduction and handoff

Start with CHANGING_STATE_001_REPORT.md. This package contains the exact original
public CSM archive, the new experiment code/data, all three new slot-store
checkpoints with optimizer states and training logs, raw predictions, timing if
admitted, and separate review-session code and receipts.

## Frozen checkpoint replay, no retraining

Use Python 3.12.14, Torch 2.5.1+cpu, NumPy 2.3.5. The exact package inventory and CPU
identity are under slot_control_001/runtime/. From this research directory:

```bash
python REPLAY_CHANGING_STATE_001.py \
  --archive ../CSM-Research-Update-20261109.zip \
  --output /absolute/path/to/new-empty-replay
```

Add `--timing` to repeat the exploratory fresh-query timing (hardware dependent).
The output directory must not exist. The helper validates the original archive's
SHA-256, restores it in that directory, copies the frozen experiment definitions
and trained slot checkpoints, and runs both frozen evaluations. It never fits
or changes original results. Runtime receipt timestamps/paths/durations will
differ; predictions and summary correctness metrics should reproduce.

A checkout can instead supply the reassembled archive at
`artifacts/csm-address-002-recovered-20261109/CSM-Research-Update-20261109.zip`.
Its README provides reassembly instructions. Original source parent is
3edc4f1a541a3221390e4c0a21fe0c12004f1ca5, branch
research/csm-recovered-address-002-20261109 in FPC-effortless/cdl-attention-experiment.

## Scope and continuation

The changing-state protocol/source/data were frozen before scoring. The slot
protocol/source were frozen and independently reviewed before fitting; all three
fits closed before either stress evaluation. The first draft closure weakness,
fix and old freeze are retained in git/review/prefit_drafts. No failed scientific
seed was dropped. Eight base stream patterns each have eight correlated value
shifts; all four identity queries per world are separate fresh-state replays.

The lifecycle interpreter covers SET, DELETE, COPY, LINK, NOP and dependent QUERY.
Neural models in this package were scored only on SET; the interpreter's successful
checks must not be presented as learned deletion, link traversal or composition.

Next executable campaign should extend the input/output alphabet and freeze a
jointly trained CSM/slot/MemNet comparison on snapshot-vs-live-reference consequences
and deletion/recreation. Preserve this SET control. Equal process labels and
answer-only tracks must stay separate. Source checkpoints cannot simply be
credited with unseen-operation competence. Fit recipes, independent private
held-out families and resource admission must be concrete before future fitting.
The current public development worlds are permanently ineligible as untouched
confirmation data.

## Publication

A GitHub push was rejected by automatic approval review because this turn did not
explicitly authorize public publication. No alternate push or API publication was
attempted. The prepared changes are on local branch
research/changing-state-001-20261113. No merge, public research claim, worker
expansion, remote deployment or paid compute is represented here.
