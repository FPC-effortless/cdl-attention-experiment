# COG-ATTN on real GitHub issue histories

**Recorded state replay passed. Automatic language-driven state changes failed the qualification gate.** The test used 2,340 actual issues across NumPy, pandas, Matplotlib, SciPy and SymPy. It paired human comments with GitHub's recorded close and reopen events. The 2024 cohort supplied training examples, the 2025 cohort selected models and thresholds, and two disjoint 2026 cohorts tested time transfer and transfer to entirely held-out SymPy. The task predicts the next same-actor state change within 15 minutes and before another human comment. GitHub records supply an observable behavioral target. Historical authority, intent and causality have no independent labels in this corpus. [Frozen protocol](PROTOCOL.md)

## Language grounding

| Frozen primary cohort | Opportunities | Observed positive transitions | Prevalence |
| --- | ---: | ---: | ---: |
| Training | 2,651 | 42 | 1.58% |
| Validation | 2,568 | 54 | 2.10% |
| Later issues in familiar repositories | 2,964 | 56 | 1.89% |
| Held-out SymPy | 533 | 11 | 2.06% |

| Ranker | Average precision, later familiar repositories | Average precision, held-out SymPy |
| --- | ---: | ---: |
| Constant score | 1.89% | 2.06% |
| History alone | 2.40% | 2.20% |
| Word and character TF-IDF | 26.39% | 15.46% |
| TF-IDF plus history | 23.21% | 14.86% |
| Zero-shot DeBERTa entailment | 26.10% | 4.24% |
| Input-matched flat fusion | 34.96% | **18.92%** |
| Modular score with NLI features removed | 35.17% | 13.58% |
| Modular text, history and entailment fusion | **40.13%** | 13.52% |

The table reports medians across three grouped cross-fitting seeds on the same corpus. The flat baselines are deterministic. The modular model's AP ranged from 39.64% to 40.54% in familiar repositories and 13.27% to 13.61% in SymPy. The modular score improved in familiar repositories when it incorporated entailment. Exact binding reuse covered 13 familiar held-out comments, all negatives, and zero SymPy comments. It changed no held-out admissions.

The paired issue-cluster bootstrap for the primary seed placed the modular-minus-flat AP difference between **−2.68 and +11.15 percentage points** in familiar repositories and **−13.84 to +3.38 points** on SymPy. Both intervals span zero. The familiar-repository AP difference for adding entailment to the modular ranker was **+1.48 to +9.69 points**. The intervals quantify within-cohort sampling uncertainty and use a single held-out repository for cross-repository inference.

The 2025 gate chose modular thresholds of 0.80 or 0.90, with 5–6 correct admissions and zero errors in validation. It selected no additional corroboration requirement. On the later same-repository cohort it proposed **3–4 correct changes out of 56**, with zero false proposals. Its best run therefore had 7.14% positive recall, 0.135% comment coverage and a one-sided 95% issue-level lower bound of 47.29%. It proposed **zero changes out of 11** in SymPy. Flat fusion proposed two changes in familiar repositories, one correct and one false, and zero in SymPy. Every policy failed the frozen joint requirement of 99% precision, 70% positive recall and a 99% issue-level lower precision bound. Even perfect prediction of the available positives could establish lower bounds of at most 94.79% and 76.16% in the two cohorts under independent issue sampling.

All 552 eligible comments sharing a timestamp with another human comment or state action had a linked transition under GitHub's server order. The 196 held-out timestamp ties therefore form a positive-only sensitivity slice. The unchanged modular policies recovered 8–11 of 158 familiar-repository ties and 3–4 of 38 SymPy ties. Primary estimates exclude those ties to preserve prospective ordering. Both strict held-out cohorts contain only close positives. Three-way ambiguity, reopening recall and unrestricted action grounding require additional evidence.

The frozen no-NLI ranking ablation removed entailment features before scoring. Its optional admission gate still consulted the entailment corroborator. The audit preserves the original artifact and separately recomputes this ablation's gate using just the explicit cue or probability. The cue-only gate recovered one validation positive per seed and admitted zero held-out examples. The primary policy and ranking used the originally frozen scores. [Audit results](audit.json)

## State and provenance

The same SQLite transaction core and WR-006 history index replayed **1,647 recorded transitions** in 2,287 usable issue histories. The implementations agreed with the prefix oracle on **9,268/9,268** observed comment checkpoints. Historical lookups also matched after all eligible transitions had been loaded. The replay excluded 53 timelines with nonalternating state actions. These exclusions condition the replay result on a valid recorded stream. Full histories are scanned retrospectively to make the selection.

Latest-stream state agreed with the separately fetched search snapshot in **2,334/2,340** cases. On all six disagreements, a direct issue snapshot again reported closed and provided a close timestamp, while both fully paginated event endpoints yielded open without any close event. All six close timestamps were inside the applicable observation windows. The mismatch persists across endpoint types that share the same GitHub authority. Event-only reconstruction consequently loses some real state. [Snapshot cross-check](snapshot-crosscheck.json)

The index stored 151,584 bytes of packed arrays. The SQLite database held 323,584 bytes of pages and a 4,140,632-byte WAL at the replay measurement point. The semantic preprocessing pass took 187.6 seconds over 9,268 comments on two ONNX CPU threads, in batches of 24 with a 128-token cap. These are local payload and offline preprocessing measurements. Deployment latency, cross-host state visibility, allocator overhead, full-process RSS and historical ingestion timestamps remain outside this run.

## Verification and handoff

Research Lab OS froze the six paired candidate/control runs. TrainLab verified all six initial runs and six clean-directory replays, with exact prediction digest agreement. A separate audit implementation recomputed all 9,268 labels directly from the saved timelines and rechecked predictions, metrics and source digests. Veritas recorded the result as an authentic-source diagnostic. The checks ran on one host with one operator and the same versioned corpus.

[Reproduction instructions](REPRODUCE.md) · [Public source issue manifest](selected-issues.json) · [Run and replay receipts](run-index.json) · [Scored audit](audit.json)

The GitHub handoff contains the source, configuration, raw-source hashes, public issue identifiers and non-text predictions. The exact archived comment bodies and trained vocabularies are retained in the restricted local evidence archive. That archive also preserves the checkpoint, all run outputs and full audit provenance.
