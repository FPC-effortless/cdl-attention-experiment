# Frozen-prediction diagnostic replay

Independent, post-fit, pure-Python replay of fixed predictions. No model calls, additional training, checkpoint selection or oracle corrections.

The original primary metrics reconcile exactly for every scored pair. This replay does not change qualification thresholds, fit another model, select a checkpoint, or apply an oracle repair.

## Definitions

A missed write predicts NOOP for a required write; an unrequested write proposes a write for a gold NOOP. Among cases where both operations request a write, address-only, value-only and joint address/value failures are mutually exclusive. These are components of the jointly chosen operation, not separate learned heads. Invalid typed proposals are reported separately and are rejected by execution, never corrected.

Attempted commitment means a non-NOOP prediction. Effective commitment additionally requires a valid record/value type. The JSON reports both confusion tables; an invalid write can be an attempted commitment without changing the ledger.

Required-chain correctness uses the gold question route and final gold link solely inside this scorer: one owner record for a box query, or the item link plus its required owner record for an item query. Every record in that chain must equal the gold final ledger. Predicted query routes and actual learned answers are assessed separately. No gold record, route or answer is supplied to the model.

Damage means a previously correct, known record whose gold value did not change at this event became incorrect. First-error and first-damage positions are zero-based and reported per history; the earliest-event index across a pair is also retained. This avoids treating world0-first traversal as temporal ordering between worlds.

## Familiar TEST

| Arm | Seed | Strict pairs | Both required chains | Chains + answers | Both routes | Both answers |
|---|---:|---:|---:|---:|---:|---:|
| control | 17 | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |
| candidate | 17 | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |
| control | 29 | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |
| candidate | 29 | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |
| control | 41 | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |
| candidate | 41 | 256/256 | 256/256 | 256/256 | 256/256 | 256/256 |

| Arm | Seed | Missed | Unrequested | Address only | Value only | Address + value | Invalid | Damaged records |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| control | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| control | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Counts are event-level; invalid proposals overlap the mutually exclusive operation-error categories. The JSON supplies the event and conditional-write denominators, first-position histograms, and length breakdowns.

## Fresh TEST

| Arm | Seed | Strict pairs | Both required chains | Chains + answers | Both routes | Both answers |
|---|---:|---:|---:|---:|---:|---:|
| control | 17 | 0/256 | 0/256 | 0/256 | 136/256 | 0/256 |
| candidate | 17 | 0/256 | 9/256 | 8/256 | 232/256 | 9/256 |
| control | 29 | 0/256 | 0/256 | 0/256 | 152/256 | 0/256 |
| candidate | 29 | 0/256 | 0/256 | 0/256 | 192/256 | 0/256 |
| control | 41 | 0/256 | 0/256 | 0/256 | 80/256 | 0/256 |
| candidate | 41 | 0/256 | 2/256 | 0/256 | 160/256 | 0/256 |

| Arm | Seed | Missed | Unrequested | Address only | Value only | Address + value | Invalid | Damaged records |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 17 | 11695 | 262 | 260 | 61 | 526 | 0 | 108 |
| candidate | 17 | 5573 | 997 | 883 | 1249 | 3050 | 0 | 698 |
| control | 29 | 10767 | 529 | 220 | 162 | 1309 | 0 | 145 |
| candidate | 29 | 4156 | 1464 | 1151 | 700 | 5719 | 0 | 834 |
| control | 41 | 12005 | 18 | 74 | 22 | 125 | 0 | 16 |
| candidate | 41 | 6037 | 989 | 1013 | 881 | 2791 | 0 | 811 |

Counts are event-level; invalid proposals overlap the mutually exclusive operation-error categories. The JSON supplies the event and conditional-write denominators, first-position histograms, and length breakdowns.

## Where fresh-history errors begin

| Arm | Seed | First event already wrong | Histories with later damage |
|---|---:|---:|---:|
| control | 17 | 460/512 | 93/512 |
| candidate | 17 | 386/512 | 260/512 |
| control | 29 | 450/512 | 88/512 |
| candidate | 29 | 464/512 | 312/512 |
| control | 41 | 394/512 | 16/512 |
| candidate | 41 | 366/512 | 245/512 |

These errors include the initial fact-establishment steps. They cannot be attributed solely to long-history accumulation. Damage counts depend on whether a history first established a correct record, so raw damage totals are not exposure-matched error rates.

## Held-out grammar breakdown

| Arm | Seed | Grammar | Strict pairs | Both required chains | Both answers |
|---|---:|---|---:|---:|---:|
| control | 17 | fresh_destination_first | 0/64 | 0/64 | 0/64 |
| control | 17 | fresh_passive_delivery | 0/64 | 0/64 | 0/64 |
| control | 17 | fresh_retention_contrast | 0/64 | 0/64 | 0/64 |
| control | 17 | fresh_status_front | 0/64 | 0/64 | 0/64 |
| candidate | 17 | fresh_destination_first | 0/64 | 0/64 | 0/64 |
| candidate | 17 | fresh_passive_delivery | 0/64 | 7/64 | 7/64 |
| candidate | 17 | fresh_retention_contrast | 0/64 | 1/64 | 1/64 |
| candidate | 17 | fresh_status_front | 0/64 | 1/64 | 1/64 |
| control | 29 | fresh_destination_first | 0/64 | 0/64 | 0/64 |
| control | 29 | fresh_passive_delivery | 0/64 | 0/64 | 0/64 |
| control | 29 | fresh_retention_contrast | 0/64 | 0/64 | 0/64 |
| control | 29 | fresh_status_front | 0/64 | 0/64 | 0/64 |
| candidate | 29 | fresh_destination_first | 0/64 | 0/64 | 0/64 |
| candidate | 29 | fresh_passive_delivery | 0/64 | 0/64 | 0/64 |
| candidate | 29 | fresh_retention_contrast | 0/64 | 0/64 | 0/64 |
| candidate | 29 | fresh_status_front | 0/64 | 0/64 | 0/64 |
| control | 41 | fresh_destination_first | 0/64 | 0/64 | 0/64 |
| control | 41 | fresh_passive_delivery | 0/64 | 0/64 | 0/64 |
| control | 41 | fresh_retention_contrast | 0/64 | 0/64 | 0/64 |
| control | 41 | fresh_status_front | 0/64 | 0/64 | 0/64 |
| candidate | 41 | fresh_destination_first | 0/64 | 0/64 | 0/64 |
| candidate | 41 | fresh_passive_delivery | 0/64 | 2/64 | 0/64 |
| candidate | 41 | fresh_retention_contrast | 0/64 | 0/64 | 0/64 |
| candidate | 41 | fresh_status_front | 0/64 | 0/64 | 0/64 |

## Evidence and limits

Reconciled **6912 pair rows** across all six frozen checkpoints and the four original panels. Every original field was compared, including first error, operation count, final ledgers, answers, routes, invalid writes and damage.

The three seeds reuse the same test families; they are head-initialization replications on one frozen native backbone, not independent language models or independent populations. Counts describe these fixed synthetic histories. A component diagnosis is not evidence that a proposed repair would succeed. There are no new model calls or counterfactual oracle corrections in this diagnostic.

The fresh families change task vocabulary as well as syntax. Their failures do not isolate word order or role binding from untrained lexical associations in the factual adapter. Native language pretraining is not claimed to exclude those words. One selected frozen pretrained backbone with three head-initialization seeds cannot qualify or refute the broader COG architecture. The original strict endpoint and qualification thresholds remain unchanged; these additional component and chain-intersection counts are post-fit diagnostics.

`diagnostic-summary.json` contains all per-arm/seed/panel/grammar/length counts and input hashes. Detailed history rows are kept with the private experimental artifacts. The corpus hash, checkpoint hashes, feature-cache receipt and prediction-export metadata are validated before replay.
