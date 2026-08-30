# CASM-Y explicit answer-state full result — 2026-08-30

CASM-Y tests a persistent proposed-answer state `y` separated from latent reasoning state `z`. Gold answer bytes are masked before encoding, answer slots are updated recurrently with shared weights, and one-step versus three-step variants have exact parameter parity.

## Run contract

- GitHub Actions run: `33283454010`
- Exact experimental head: `4db686bfd4fff498c50a953b29be193a4d022f01`
- 2,000 paired training steps
- Parameters: 1,556,801 for both variants
- Variants: `answer-state-1step`, `answer-state-3step`
- Answer targets: bytes + EOS
- Primary endpoint: exact generated answer
- Graph guardrail: exactly balanced `yes` / `no`

## Internal held-out summary

| Metric | one-step | three-step |
|---|---:|---:|
| Easy answer NLL | **1.4409** | 1.4519 |
| Easy exact | 19/256 (7.42%) | **22/256 (8.59%)** |
| Hard answer NLL | **1.6295** | 1.6506 |
| Hard exact | **28/256 (10.94%)** | 27/256 (10.55%) |

For the three-step model, hard step NLL changes only slightly: 1.6652 -> 1.6511 -> 1.6506. Recurrence is therefore not producing a strong progressive solution refinement signal.

## Hard six-task exact solves

60 held-out hard examples per task, 360 total:

| Task | one-step | three-step |
|---|---:|---:|
| Associative recall | 2/60 | 1/60 |
| State tracking | 0/60 | 0/60 |
| Arithmetic | 1/60 | 0/60 |
| Rule induction | 0/60 | 0/60 |
| Graph reachability | 29/60 | 29/60 |
| Reverse | 0/60 | 0/60 |
| **Total** | **32/360 (8.89%)** | **30/360 (8.33%)** |

For comparison, the earlier full CASM-D deep-answer treatment reached 39/360 (10.83%). CASM-Y therefore does not improve the main solve-rate endpoint.

## Exact-balanced graph guardrail

Both variants predict `no` for all 200 balanced graph examples across easy and hard splits:

- easy: 50/100 = 0/50 `yes`, 50/50 `no`;
- hard: 50/100 = 0/50 `yes`, 50/50 `no`.

The natural graph exact counts are therefore class-prior artifacts rather than learned reachability.

## Long-horizon stress

All tested long-horizon state and associative groups are 0/20 exact for both variants:

- state: 12, 24, 48, 96 events;
- associative recall: 12, 24, 48, 96 keys.

The generated state answers remain malformed or class-like strings such as `boo`, `roo`, and related mixtures.

## Effective recurrence depth

Using the trained three-step checkpoint on 180 hard examples:

| Depth | Exact | Answer NLL |
|---:|---:|---:|
| 1 | 16/180 (8.89%) | 1.6083 |
| 2 | 16/180 (8.89%) | **1.5964** |
| 3 | 15/180 (8.33%) | 1.5971 |
| 5 | 15/180 (8.33%) | 1.6201 |

Additional recurrent compute does not improve exact solving and eventually degrades likelihood.

## Conclusion

**CASM-Y is rejected in its current form.** Separating latent reasoning `z` from a persistent continuous answer state `y` is not sufficient when `y` is represented as unconstrained parallel output slots. The model learns answer marginals and class priors rather than task-dependent structured solution revision.

This narrows the viable recursive-answer hypothesis: any successor should make `y` a structured candidate solution with explicit internal dependencies or edit/update semantics, rather than independent continuous answer positions.

A parameter-comparable ordinary full-context Transformer control is being run at the same 2,000-step budget before attributing the remaining failure specifically to CASM's compressed state architecture.
