# COG-INFO-001: query identity survives; another readout did not recover the weak errors

This audit continues nanoCOG's original local binding repair. It is not
a new backbone plan and does not reopen bounded retention or main
training. See CONTINUITY.md for the preserved workstreams.

Two exact frozen ROLE-002 checkpoints (1171 strong and 1371 weak).
192 fresh factorial families, excluding 1,144 previously used joint
combinations; 128 probe-training families and 64 held families grouped
by **disjoint unordered name pairs**. All 16 names and eight values
are familiar vocabulary. Two independent native trace and two native
terminal-logit observations, no change to the original models. Separate
declared fixed logistic probes use the captured t=6 feature and normalized
literal query. No candidate colors as probe inputs.

| On the same 512 held worlds | Strong 1171 | Weak 1371 |
|---|---:|---:|
| Literal identity, normalized query-only probe | 512 | 512 |
| Rendered-role probe, q + L2 gate input | 510 | 463 |
| Rendered-role probe, q + gate probabilities | 511 | 490 |
| Selected-value probe, q + L2 gate input, 8 colors | 250 | 200 |
| Selected-value probe, q + pre-output hidden, 8 colors | 510 | 433 |
| Native original readout, strict full 287 IDs | 509 | 434 |

Query identity can be recovered in both, as expected from the literal
embedding. Before the output head, the chosen probe can also decode
much of the role signal from the weak model's gate. That is not a
proof that the native model applies the role correctly. Gate probability
mostly reports a selection decision and does not by itself carry the
value content. Some feature points are nonnested, so probe accuracy
need not be monotonically increasing across them.

## The matched readout check

At the **same 512 held rows**, repeat native decoding and require the
entire 1,536-row pre-output state tensor to be byte-identical to the
sealed earlier trace for that checkpoint. Score native separately
against 287, eight, and the same two visible colors; compare the probe
only within matching eight- and two-color spaces. Native ranks happen
to agree on this particular panel. Both matching comparisons agree:

- Strong: 509 native → 510 probe; one second-clause rescue, no damage.
- Weak: 434 native → 433 probe; **zero rescues**, one second-clause damage.
- Weak eight-color CE decreases from 0.3773 to 0.3643 nats, and
  two-color CE from 0.3373 to 0.3155. Lower log loss does not equal
  more correct choices.

Replacing the native decision with this augmented linear readout does
not yield the missing accuracy. That weakens the case for prioritizing
a simple head replacement and motivates a causal test of gate and
value-composition paths. It does not prove exactly where usable
information is lost or exclude a different readout. No 8-color probe
is credited as a 287-ID model.

The within-family label shuffle preserves each family's color/name set,
so it is not a chance baseline: eventual value top-1 can still be
469/512 strong and 402/512 weak, with much worse eight-color CE
(0.8969/1.0040 nats versus 0.0295/0.3643). The probe CE is an estimate
from this fixed held sample and an upper bound on corresponding
conditional entropy only at population level under the same distribution,
not an estimate of mutual information or proof of absence. Declared
bootstrap grouping is by the 16 held name pairs. No row-IID claim.

## Custody and continuation

Trace and readout runs: two native forward-only jobs each, 768 forward
minibatches total, zero upstream model parameter updates; only the
separate small fixed diagnostic probes learned readout weights. No
protected TEST, MAIN or paid compute. Same local gates, fixed-eight-slot
retention tests and matched-quality end-to-end inference comparisons
remain on the original roadmap. The next prospective INFO-002 test uses
matched queried-value/distractor edits on fresh cases and a controlled
gate/tap factorial within each checkpoint, not an active-inference
controller.

- Trace seal: `f868706de68fd723e0e87388969468763cd9274d5fae4739d173cb6107f262df`
- Probe aggregate: `797243b7e0cf8c118ccbdb4a68b615f15290e2f36e09c96574a6cb1680651c8d`
- Independent probe audit: `92b4f017cea364ed63b7b6959e488dba7e9d21a03b4f370bbe9e3624b919da99`
- Matched native replay seal: `b8d2bbb188bfb389a39026a5f05a6d79e779212606ec63fb8420397463721fd4`
- Comparison: `b454964f874de06e73fc0500b924797f54bf0acbd94831dd39aff78db2ee4ffb`
- Independent native comparison audit: `ccd6e8f17698f00a826778639d0c405bc25099950a5827398fd0559c8e165fdf`
