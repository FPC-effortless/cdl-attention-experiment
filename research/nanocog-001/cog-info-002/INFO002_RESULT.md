# COG-INFO-002: the value path changes answers; the gate edit alone does not

**The preregistered causal probe passed. The nanoCOG local binding gate did not change.**
INFO-002 follows INFO-001's negative final-readout result within the same
nanoCOG local learnability program. Two exact frozen gated32 ROLE-002
checkpoints ran 416 one-thread forwards each, zero updates, on 32 fresh
name-pair groups with eight fully crossed receiver worlds per group and
two matched one-token edit donors per world. The 96 exact name/color
family keys exclude 1,336 previously used keys. Names, syntax and
backbone are familiar; the groups share name identities.

## What changed

For one receiver, keep its query, residual and weights fixed. On just
the final token in layer 2, replace only the four gate scores (G), the
four contextual tap vectors those gates mix (U), or both with the donor's.
The matched control edits the other person's value in the assignment-
swapped sibling, using the same ordered old→new color change for each
target comparator. Recompute the unmodified local operator and original
downstream network. All 384 earlier-output checks on hybrid forwards, 192 exact same-row forward checks,
64 hook-removal replays, and 576 recomputations passed FP32 bitwise
checks. No gold answer enters the model. The scorer requires the unique
maximum over all 287 tokens; a forced two-color choice does not count.

## Registered weak checkpoint, query about the second clause

| Change | Correct new answer / 128 target edits | Crossed target-minus-distractor logit-margin change |
|---|---:|---:|
| Unedited receiver (scored against new value) | 0 | Reference |
| Gate scores only | 0 | −0.005 logit units |
| Contextual taps only | **93** | **+6.183 [4.672, 7.582]** |
| Gate scores and taps | 94 | +6.198 |
| Natural edited input | 94 | +6.253 |

The interval is the registered 2,000-draw group bootstrap, descriptive
of this selected panel, not cross-seed or cross-language uncertainty.
Tap transfer recovered the new donor answer in 93 rows with no
donor-gold losses. It also chose the inserted **distractor** in 33/128
rows, 3 of which harmed an originally correct receiver; three originally
wrong receivers instead became correct. Original-answer accuracy on
distractor edits therefore stays 95/128, but row-level errors move.
The strong checkpoint's corresponding tap-only condition achieved
127/128 target edits and inserted the distractor in 0/128. First-clause
tap edit accuracy was 124/128 weak and 126/128 strong (descriptive).

## Interpretation

Changing what reached the final mixer changed what the frozen weak
model answered, even with its gate scores and output head untouched.
Changing only the gate scores for these same value edits did not
reproduce that change. Both a misdirected value effect and incomplete
target binding remain in the weak receiver; the natural edited-input
control itself gets only 94/128. Thus another gate-only edit or bigger
memory is not the best bet from this result. The next targeted repair
should change how contextual values represent the queried relationship
and require sensitivity to the target *and* invariance to a matched
distractor. Keep the original matched controls and qualification floors.
The local, retention/routing, and end-to-end inference stages remain
on the same roadmap; no MAIN or protected TEST was opened.

This is a one-direction color insertion and a hybrid intervention, not
proof of natural mediation, uniqueness, stable syntax transfer,
successful training or an inference advantage. The 93 edited answers
are not rescues of 93 original native errors. The 33 inappropriate
distractor answers include rows already wrong before intervention.
Exact Successor Lookup, persistent memory and the attention replacement
claim were not tested by this probe.

Reproducibility: the frozen protocol, actor and scorer are recorded
here; raw logits, exact weights, admissions, and replay material stay
in the owner's private archive. Source/aggregate publication is not
publication of private checkpoints or datasets.
