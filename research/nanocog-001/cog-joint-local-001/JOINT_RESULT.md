# COG-JOINT-LOCAL-001: end-to-end training shifted the role errors

**No arm passed the held local-binding gates.** All nine fits ran under the
fixed schedule, but only four met the separate TRAIN adequacy check. The
result is a bounded failure of these training recipes, not evidence that
either mixer architecture is inherently unable to bind relations.

## What ran

Three 37,414-parameter, seven-token causal arms were trained end to end
from fresh paired seeds: gated32 with mean edit loss, gated32 with a
fixed 50:50 mean/log-mean-exp role-tail loss, and four-tap causal
attention32 trained with that same role-tail loss. The gated pair
shared exact initialization, schedule, answer supervision, prefix
loss, edit margins, and data. The attention arm used corresponding
shared-trunk initialization and equal training exposure, not presumed
equal FLOPs. Each fit took exactly 1,280 updates; both fixed endpoints
(128 and 1,280) and the old frozen native reference were scored.
Nineteen actors read only opaque prompt projections; the scorer used
the separate truth projection. No endpoint selection.

The adapter-free study held out 32 edit name-pair groups (128 base
histories, 128 triplets per queried clause) and 120 ordinary all-eight
families (960 questions). These were disjoint from the jointly trained
name-pair pool and excluded 1,824 exact prior family keys. The raw
language remains a familiar controlled grammar, not a natural-language
transfer benchmark.

## Final held counts

A triplet means that the model gives the right original answer,
follows an edit to the queried entity, and preserves the answer when
the other entity is edited. All three outputs must be unique full
287-token maxima.

| Arm / seed | First query triplets / 128 | Second query triplets / 128 | Ordinary worlds / 960 | Complete families / 120 | TRAIN adequacy |
|---|---:|---:|---:|---:|:---:|
| Frozen native | 126 | 98 | 839 | 63 | Reference |
| Gated mean 2371 | 82 | 128 | 782 | 54 | No |
| Gated tail 2371 | 82 | 128 | 772 | 47 | No |
| Attention tail 2371 | 89 | 126 | 801 | 55 | Yes |
| Gated mean 2471 | 48 | 75 | 580 | 0 | No |
| Gated tail 2471 | 59 | 76 | 584 | 0 | No |
| Attention tail 2471 | 79 | 128 | 726 | 54 | Yes |
| Gated mean 2571 | 90 | 128 | 859 | 81 | Yes |
| Gated tail 2571 | 89 | 128 | 858 | 80 | Yes |
| Attention tail 2571 | 69 | 127 | 810 | 59 | No |
| Required absolute floors | ≥116 | ≥116 | ≥912 | ≥108 | See protocol |

Every learned run lost first-query triplets versus the frozen reference.
Ordinary first-clause correctness fell from 472/480 native to 214–379/480
among the trained arms. Seven of nine trained runs cleared the
second-clause triplet floor, but none cleared the first. Wrong first-query
distractor adoptions rose from 0 native to 28–51/128 across learned runs.
The gated tail's paired second-query changes were 0, +1, and 0
triplets versus gated mean (the two 128/128 means had no room for
a seven-triplet gain). There was also no complete-family or cross-role
gate pass masked by that ceiling. No arm matched the frozen reference's
97/128 complete same-history four-answer cases; the best was 91.

## Decision

Stop this fixed loss comparison rather than extending training on the
exposed held cohort or choosing a favorable seed. The result shows a
role tradeoff under these recipes: gains on the second clause did not
coexist with first-clause robustness. In particular even both
TRAIN-adequate attention seeds failed held first-clause and ordinary
gates, so there is no demonstrated architectural winner. The separate
TRAIN adequacy check prevented attributing poor outcomes to architecture
when some fits had not stabilized. Do not infer the cause of the
role tradeoff from output counts alone.

A subsequent protocol should preserve this exposed panel as a diagnostic
and require both roles to transfer together on new, stricter cohorts.
It must earn any change to representation or training budget through
prospective TRAIN adequacy, not through post-hoc edits to these gates.
Fixed-slot retention and full-cost matched-quality inference were not
opened. No MAIN or protected TEST use, promotion, or deployment.

## Replay

Native TrainLab receipts cover 11,520 updates across nine fits, four
separate TRAIN-only evaluator rehearsals, and all 19 blind evaluations.
The only learning claim here is the measured controlled behavior; process
durations are not a serving benchmark. The model-free scorer reads the
sealed full logit wires and produces the aggregate results. Key identities:

- Fit ledger SHA-256: `061865dab372fae832d4c544d82c77f934d0459eaa2fda7bccf08367807c16a7`
- Held plan SHA-256: `2e1d5dcfa332f5c11536f9fbe1b62822dce3e9955aface958b8bb055261d89ff`
- Held seal SHA-256: `6d47358658fa9037427427a763ef58a90909d386e13c7197849b193eadc2016c`
- Score SHA-256: `ef20b2a45b95038ae22edc7e4cd313df9f5ff1e1a81ec16771cfebd45d423293`

Exact weights, cohorts, receipts and raw wires remain in the private
replay archive; public files contain source and aggregate results only.
