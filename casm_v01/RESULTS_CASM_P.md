# CASM-P — Full 2,000-step result

Status: **negative for the promotion criterion**.

CASM-P tested whether a training-only process decoder could align the three recurrent reasoning states with deterministic verifier-generated intermediate computation traces. The deployed core remained the same 1,347,361-parameter recurrent Q/K model.

## Primary result

The auxiliary process objective was highly learnable (process-state cosine reached approximately 0.97), but this did not translate into a material increase in true autoregressive problem solving.

Across 360 hard free-generation examples (60 per task):

- final-only-3step: 37 / 360 exact solves (10.28%)
- process-3step: 38 / 360 exact solves (10.56%)

The treatment therefore added **one exact solve out of 360**, below the predeclared promotion threshold of a meaningful solve-rate improvement.

Per task, the process model solved 7/60 state-tracking examples versus 6/60 for the control; both solved 31/60 graph examples, and both solved 0/60 associative recall, arithmetic, rule induction, and reverse/copy examples in this free-generation evaluation.

## Likelihood metrics

Separate corrected six-task evaluation:

| model | overall answer NLL | answer-byte accuracy | teacher-forced exact |
|---|---:|---:|---:|
| final-only-3step | 1.472332 | 0.423733 | 0.113889 |
| process-3step | 1.471882 | 0.425866 | 0.118056 |

The differences are negligible and do not change the solve-rate conclusion.

The packed hard evaluation from `summary.csv` was similarly close:

| model | hard answer NLL | hard answer-byte accuracy |
|---|---:|---:|
| final-only-3step | 1.330466 | 0.490396 |
| process-3step | 1.322509 | 0.491423 |

## Exact-balanced graph falsifier

On exactly 50 reachable and 50 unreachable examples per difficulty, neither model learned graph reachability.

- final-only easy: 50%, predicts `yes` for every example.
- final-only hard: 50%, predicts `yes` for every example.
- process easy: 50%, predicts `no` for every example.
- process hard: 50%, predicts `yes` for every example.

Thus the graph result is **class collapse**, not reasoning.

## Long-horizon generation

Long associative-recall generation remained 0% exact for both models at every tested horizon. State-tracking free-generation performance was low and mixed; process supervision did not produce a systematic improvement.

## Interpretation

CASM-P demonstrates that the recurrent state can be made highly decodable by a separate training-only MLP without forcing that computation to become useful to the model's own output geometry. This formulation is rejected.

The next ablation should remove the auxiliary process decoder and apply **direct deep supervision through the shared LM head** at intermediate recurrent steps, keeping the deployed parameter count unchanged. This is tested by CASM-D.
