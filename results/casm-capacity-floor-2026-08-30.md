# CASM single-task capacity floor — 2026-08-30

This diagnostic asks whether the existing ~1.35M recurrent CASM core can learn corrected state tracking or graph reachability when multi-task interference is removed.

## Run contract

- GitHub Actions run: `33282252481`
- Branch: `diagnostic/casm-capacity-floor`
- Two independent specialized models
- 1,200 training steps per model
- Same tiny recurrent CASM family used in the multi-task experiments
- State prompts include explicit initial state
- Graph evaluation is exactly balanced between `yes` and `no`
- Primary endpoint: true generated exact answers, not teacher-forced loss

## State tracking

Training LM loss falls to roughly 0.22–0.24 by step 1,200, but generated solution accuracy remains low:

| Split | Exact solves |
|---|---:|
| Easy | 16/100 (16%) |
| Hard | 6/100 (6%) |

Easy class detail:

- `bag`: 3/18
- `box`: 13/27
- `desk`: 0/11
- `room`: 0/9
- `shelf`: 0/16
- `tray`: 0/19

Hard detail includes only 1/12 `bag` and 5/17 `box`; all other location classes are 0. Generated outputs frequently collapse to common labels or continue into prompt-like fragments.

## Graph reachability

The graph-specialized model reaches training LM loss around 0.41, but exact-balanced generation remains pure class collapse:

| Split | Overall | `yes` | `no` |
|---|---:|---:|---:|
| Easy | 50/100 | 0/50 | 50/50 |
| Hard | 50/100 | 0/50 | 50/50 |

The model predicts `no` for every balanced graph example.

## Conclusion

**The multi-task curriculum is not the main bottleneck.** Removing task interference and dedicating 1,200 steps to one task does not yield robust state tracking or graph reachability.

The diagnostic also reinforces a measurement rule for the project: low teacher-forced/full-sequence LM loss can coexist with near-absent solution competence. Future promotion gates must therefore use exact generated solutions and class-balanced task metrics wherever possible.
