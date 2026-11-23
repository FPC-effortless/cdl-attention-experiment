# COG-BIND-001P: the improved route survives the rewrite; its rejection gate still over-refuses

## Result
The three frozen diverse 001O routes got **4,992/4,992** valid familiar answers and **4,982/4,992 (99.8%)** valid rewritten answers exactly right, including their directed supporting record. With familiar wording, both learned gates preserved every correct valid answer and passed both prospectively declared screens in all three seeds. Under the unseen rewording, **neither gate passed the overall screen in any seed**. The frozen route still knew the right answer in nearly all cases; the gate mostly refused valid answers. This identifies a different bottleneck from the low native ceiling in 001N, but 001N and 001P used different datasets and this is not an apples-to-apples numerical comparison.

## Fixture and integrity
Fresh synthetic TRAIN/DEV/HELD used 256/64/128 semantic families (41 rows per family). The held novel wording is paired to the same canonical facts, not a new independent sample. Both 725-parameter gate heads use the same frozen native decisions, paired initializations and matched training exposure (10,496 queries, 32 passes, 1,312 updates per fit). Only the heads were fit, using role-separated TRAIN labels. Both heads use 12 common rank gaps; their additional inputs are either confidence/score summaries or frozen role/eligibility/pair features. Neither gets the true bridge, parser records, canonical candidate counts, or reference answer; it can only allow or reject the frozen answer. Three original diverse 001O checkpoints were used without backbone optimization.

The hard crossed panel places an equivalent repeated declaration and a genuinely conflicting alias in the same passage; which one is relevant depends on the query. Changing whether two people received the same item does not resolve a person-level ambiguity. Equivalent same-scope repeated declarations retain one canonical referent and can be credited with either witness. Distinct transfers are not merged because their items agree. Whole word bags, query-only input, context-only input and the global count of literal duplicates each have a 50% majority ceiling on this crossed panel, not a general ceiling on the entire benchmark. A first dataset draft was held before any model use when a global duplicate-line shortcut was discovered; this corrected matched panel is the only one used.

## Held results across three checkpoints
Per wording view these are 4,992 valid and 10,752 invalid model–row observations over 128 underlying held families.

| Learned gate | Familiar valid accepted and exact | Familiar invalid accepted | Rewritten valid accepted and exact | Rewritten invalid accepted |
| --- | ---: | ---: | ---: | ---: |
| Confidence summaries | 4,992 | 71 | 3,193 (64.0%) | 6 |
| Pair evidence | 4,992 | 25 | 3,628 (72.7%) | 12 |

Every learned accepted valid decision was exact in these panes. The pair head improved rewritten valid coverage, but admitted more invalid inputs than confidence alone, so it did not dominate under both objectives. The additional accepted valid cases came from one seed (pair minus confidence: −15, +671, −221). Strict favorable decisions across all 41 rows were 368/384 and 377/384 familiar, versus 53/384 and 63/384 rewritten, for confidence and pair respectively. Familiar learned gates passed the main and crossed thresholds on all three seeds; the rewritten main threshold passed none, and the pair head passed the separate rewritten crossed threshold in one seed only. No hand-coded strict margin met DEV eligibility at 90% valid coverage because equivalent duplicate bridges often tie; its declared fallback was reported, not tuned on held.

### Equal-item and redundant-evidence controls
Each conflict/item cross cell has 1,536 model–row observations per view. For equal-item alias conflicts, false accepts were 26 confidence / 10 pair on familiar wording and 4 confidence / 6 pair on revised wording. For different-item alias conflicts they were 28 / 10 and 2 / 6, respectively. Both accepted and answered all **3,072** equivalent duplicate-declaration rows exactly on familiar wording. On novel wording, accepted exact duplicates fell to **2,030/3,072** and **2,412/3,072**. Both rejected all 1,920 core conflicting-incoming cases per wording, including 960 equal-item distinct transfers. Equal item did not systematically override the configured conflict distinction in this panel; neither method is a guarantee of semantic grounding.

## Verification and next
Six gate-only fits, 12 sealed feature bundles, 18 blind gate prediction bundles, and an independent raw recount. The original scorer reused a loop variable for the predicted item; the sealed raw trace and the independent recount caught it. A reviewed scorer-only rename restored the declared native exact calculation. The original scorer and misbound score are labeled and preserved; no weights, inputs, thresholds, predictions, or training were repeated or changed.

Next, replace the black-box allow/reject inference from confidence with a check that explicitly compares canonical candidate sets for consistency and sufficiency. Keep the learned route unchanged as a control, treat any exact grammar parser as a reference only, and test rejection, answered coverage and total cost on new wording and evidence families. This is not a natural-language, deployment-safety, or inference-cost result.

Data pin `987fe0da3da2a6a58c0d9cbe1e77d8a5b28e5fa24a1a8f1af86a33c3d4c22921`; corrected score `b1ce8dad6b47566b3e9bc36fc6bf82bbb54237871585452eb565c19af653f829`.
