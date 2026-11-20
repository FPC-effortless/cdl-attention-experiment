# Follow-up on naturally occurring state changes

The earlier GitHub test showed two distinct limits. The cautious language policy admitted too little, and some complete API timelines disagreed with GitHub's current-state snapshots. This follow-up tests two repairs on newly collected, naturally occurring issue records.

## Corpus and freeze

Training and calibration use exactly the frozen 2024 and 2025 data and pinned entailment model from the first experiment. No previous test observation enters fitting or threshold selection. The new holdout comprises publicly commented issues created July through September 2026, with observations censored on November 10. Select the earliest 100 eligible issues per repository, or all eligible issues if fewer. Use four familiar Python-library repositories and a separate new-repository stratum. Three new repositories were added after search returned only 171 eligible issues in the original external stratum. The extension depended on cohort sizes alone and was frozen before fitting or inspecting fresh target labels. All eleven repositories are specified in the design file.

A strict language observation uses the original behavioral definition. An unedited human comment precedes a close or reopen by the same actor, within 15 minutes and before any intervening human comment or state event. Exclude comment-time ties and the original sequence-integrity exclusions. An auxiliary training observation is an authentic positive where GitHub places a same-actor state transition immediately after the comment at the same second. Treat these as weak behavior labels. A simultaneous close and comment can be a single UI action. Later fresh test labels and current-state snapshots stay out of prediction features. The score addresses observed behavior in a closed action space.

## Language experiment

Keep the original word and character text features, history features, entailment model, and cutoff grid. Evaluate the following comparisons on strict fresh comments.

- Strict flat and strict modular controls use only the forward-ordered training labels.
- Pooled flat and pooled modular arms also treat the same-second training positives as primary positives.
- The candidate learns a separate tied-event text expert against the same strict negatives. It adds its probability to the original separate text and state experts. The final stack is trained only on the strict out-of-fold rows. Grouped four-fold cross-fitting withholds every comment from the same issue from all experts when producing a stack training row.

Regularization and confidence thresholds use the original strict validation split. The auxiliary lexical expert has fixed C=3. Require 99% validation precision for any nonempty admission rule. Retain both point precision and one-sided 95% lower bounds over admitted issues, with any wrong action making an issue a failure. Joint qualification requires at least 99% point precision, a 99% lower bound and 70% positive recall on each fresh stratum. Report AP, recall, false proposals and one-sided bounds separately. Primary AP comparison is candidate versus pooled modular with paired issue-cluster bootstrap confidence intervals. Strict modular and the best flat references are mandatory secondary comparisons. Run three fixed cross-fitting seeds and same-host clean directory replays. The seeds share one corpus.

## Snapshot-aware runtime experiment

Use naturally occurring timeline events. The direct issue endpoint supplies a separately fetched current-state observation for scoring. Compare timeline-only folding, search-snapshot-only read, disagreement-based abstention, and a guarded close-time anchor. The guarded path may assimilate a close anchor only when GitHub's search snapshot says closed, the recorded current state disagrees or lacks the matching terminal close timestamp, and the reported close time is within the issue lifetime and strictly later than all observed state actions. An open snapshot supplies no reopen time, so that conflict stays unresolved. Every repaired close retains snapshot provenance and carries no fabricated event actor, identifier or independent authority. Nonalternating streams are recorded separately. A current-state check cannot certify complete intermediate history.

Report search/direct agreement and changed record versions. The stable subset requires exact agreement of both endpoint snapshots on current state, terminal close time and issue updated time. Report both the full collected cohort and this subset. Authorities share GitHub, with separate request paths. Re-fetch the event-only endpoint for each flag and a deterministic subset of unflagged issues. Record disagreements and preserve the original timelines.

Use the existing WR-006 bitemporal index with the original event dates and a separate observation coordinate. Insert observed history at the replay observation phase and any accepted anchors at the later repair phase. These phases are ordered program steps. The API provides no original notification or ingestion history. Compare both indexed answers against a separate scan using only facts visible by each phase. Validate the assimilated state and date against the independent direct response. The endpoints do not expose an independent ground truth for unrecorded intermediate history. Shadow replay performs no live GitHub writes.
