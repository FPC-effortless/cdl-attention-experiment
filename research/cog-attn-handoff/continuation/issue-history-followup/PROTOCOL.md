# Snapshot reconciliation and separate same-second supervision

Freeze the source selection, endpoint roles and language analysis before scoring fresh labels. Training and calibration use the earlier 2024 and 2025 issues with the exact entailment checkpoint. Hold out newly acquired issues created July 1 through September 30, 2026. Use 100 earliest publicly commented issues per project or the full eligible set if smaller. Censor label observation on November 10, 2026 UTC. Keep the four familiar repositories and the seven external repositories separate. Three external repositories extended the original external search after its first four repositories returned only 171 eligible issues. The amendment used search counts before any fresh target analysis.

| Cohort | Repositories |
| --- | --- |
| Familiar | numpy/numpy, pandas-dev/pandas, matplotlib/matplotlib, scipy/scipy |
| External | scikit-learn/scikit-learn, scikit-image/scikit-image, python-pillow/Pillow, statsmodels/statsmodels, python/cpython, jupyterlab/jupyterlab, pytorch/pytorch |

## Language grounding

Retain the strict target. An unedited human comment precedes a same-actor close or reopen within 900 seconds, before the next human comment or state action. Exclude same-second and inconsistent-history examples from strict scoring. Use naturally occurring same-second pairs as weaker auxiliary observations. GitHub can emit a comment and close as a single operation.

Compare three ways of training. The original controls learn from strict labels. Pooled controls treat same-second positives as regular positives. The candidate learns a separate text expert for same-second actions and fits its final stack on strict data using out-of-fold expert scores. Use identical pinned entailment, input features, and grouped four-fold splits. Train auxiliary logistic regression with C=3. Evaluate the strict and pooled flat baselines as well as the modular controls. Use seeds 67003, 67009 and 67021 with one shared corpus.

Select thresholds on the archived strict validation split at 99% empirical precision. The joint gate also requires 70% recall and a one-sided 95% issue-level lower bound of 99% on both fresh holdouts. Report denominators, false proposals, and average precision. Compute 1,000 paired issue-cluster bootstrap draws for the candidate-minus-pooled ranking difference. Record all runs and matched clean-directory replays in Research Lab OS and TrainLab.

## State reconstruction

Read full paginated GitHub timelines and separately fetch direct issue snapshots. Compare timeline-only, snapshot-only, disagreement abstention, and guarded snapshot-close assimilation. Snapshot inputs and the direct verification response have separate retrieval paths and share the same producer. Report the full cohort and the subset where the search and direct state, close timestamp and updated timestamp agree. Re-fetch the event-only endpoint for flagged cases and a deterministic sample of unflagged cases. Preserve conflicting or nonalternating histories.

Use the same WR-006 bitemporal index with original dates as valid time and replay observation and repair phases as knowledge time. Check answers with a separate scan. Validate the assimilated state and date against the direct response. Historical ingestion times and truth for intermediate periods are unobserved. The local prototype and the fresh replay operate in shadow mode.
