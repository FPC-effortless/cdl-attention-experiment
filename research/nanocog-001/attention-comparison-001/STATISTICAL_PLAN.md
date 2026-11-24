# Statistical review of the prospective attention comparison

**Status:** Independent prospective design critique. This note contains no experimental observations, performance scores, or scientific acceptance decisions. Read-only inspection of `source_snapshot/veritas_comparison_design.py` confirms the mechanical constraints and estimators described below. No native module import, campaign execution, or repository mutation was performed.

## 1. Admission decision

The existing comparison API is a useful bounded implementation, but passing its input checks does **not** establish adequate statistical power, sufficient independent training seeds, or reliable confidence-interval tails. A confirmatory campaign must remain unadmitted until its statistical implementation and resource plan support the intended design.

Two acceptable paths are:

1. Validate a statistically adequate implementation for the full preregistered comparison, including its resource use, paired dependence structure, and uncertainty calculation.
2. Preregister computationally manageable per-primary-family analyses, retaining every planned observation and the full multiplicity correction through an explicit allocation of the overall error budget.

Neither path permits silently removing seeds, downsampling test families, reducing bootstrap draws below a validated precision requirement, dropping unfavorable comparisons, or relabeling one campaign as several independent campaigns to escape multiplicity. An explicit smaller study designed **before** confirmatory data are examined is permissible only if its power, claim scope, and remaining limitations are reassessed and documented.

## 2. What the current API permits—and does not prove

The supplied `freeze_design` interface has these bounds:

| Quantity | Limit or condition |
|---|---:|
| Evaluation units | At most 1,000 |
| Comparisons | At most 32 |
| Total cells across comparisons: units × seeds × comparisons | At most 20,000 |
| Bootstrap draws | At most 10,000 |
| Total cells across comparisons multiplied by draws | At most 5,000,000 |
| Interval-inference eligibility | At least 2 units and 2 seeds; this is not a power safeguard |
| Adjusted-tail sampling for interval inference | Requires only at least 1 expected tail draw |

Its reported method is a Bonferroni-adjusted paired crossed unit-seed percentile bootstrap. The pairing and crossed structure are appropriate design concerns, but their implementation still needs validation. The admissible minimum of two seeds is not a scientific recommendation. Percentile intervals can be unstable or have poor coverage with few seed clusters, rare errors, or metrics near a boundary; more examples within the same seed do not repair a shortage of independent trained models.

The code computes `n = units × seeds × comparisons`; comparisons count toward **both** resource limits. A valid illustrative design with 1,000 units, 5 seeds, and 4 comparisons has 20,000 total cells and allows at most 250 draws under the 5,000,000 product limit. The verified two-sided tail convention is `alpha / (2 × comparisons)`: at alpha 0.05, the tail probability is 0.00625 and 250 draws yield only 1.5625 expected tail draws. Such a complete, fixed design can pass the mechanical tail check while leaving its interval tails poorly resolved. With 1,000 units, 5 seeds, and 32 comparisons, there are 160,000 cells, so the design is outright inadmissible even before the draw-product check. One expected tail draw is a computational minimum, not a satisfactory precision criterion.

Before admission, document the verified two-sided interval convention, its relationship to the planned noninferiority error budget, the effective number of statistical claims, empirical coverage under plausible null/boundary scenarios, and Monte Carlo uncertainty of the reported interval endpoints. Check that results are stable to bootstrap random seed and a justified increase in draws within the validated implementation. Do not treat agreement of rounded point estimates as interval validation. Expensive draws can be streamed or otherwise computed efficiently, but a new implementation must preserve the intended estimator and pass verification; bypassing a resource check without that work is not an acceptable remedy.

## 3. Unit of analysis and dependence

A unit is an independent task/document family, not a token, paraphrase, or individual history within a counterfactual pair. Keep all dependent variants together. A counterfactual family's primary outcome should include both-histories-correct success where that is the scientific target. Cluster related repository tasks, source documents, templates, and generated variants at the highest relevant shared level.

All model variants should evaluate the same fixed held-out units. Match training runs by the planned data-order seed and experimental schedule where useful; different architecture shapes do not thereby receive identical initializations. Training seeds and evaluation families are crossed: every trained seed evaluates the same families. Resampling must preserve candidate-baseline pairing and this crossed structure, rather than separately shuffling candidate and baseline observations or treating all cells as independent.

For NLL, freeze token-weighted or document-weighted aggregation in advance, preserve document clusters, and report NLL differences before exponentiating to perplexity. For latency, use repeated measurement blocks and paired workloads, with machine/session variation represented. Thousands of timed tokens in one session are not thousands of independent hardware replications.

### Estimator compatibility is a separate admission gate

The inspected `analyze` implementation computes an **equal-cell mean of candidate-minus-baseline scores**, and its bootstrap recomputes that equal-cell mean after crossed resampling. Its internal binary denominator preserves floating-point input values; it is not a token-count or sampling-weight denominator.

A primary natural-language endpoint defined as token-weighted NLL therefore needs a validated clustered **ratio-of-sums** estimator or adapter that carries each family's summed negative log likelihood and target-token count through every bootstrap resample. Average seeds according to the frozen estimand. Feeding unequal-length document means into the present function and labeling the result token-weighted NLL is incorrect. An equal-document-mean endpoint is a legitimate different estimand only if explicitly chosen before testing. Splitting documents into tokens to fit the existing equal-cell estimator would destroy the required dependence structure and is not an acceptable adapter.

Cost ratios also need an explicitly validated estimand. Use matched workloads within paired measurement-session blocks, specifying whether the target is a ratio of arithmetic means, ratio of medians, or a geometric mean of paired ratios. A validated log-cost transformation followed by exponentiation of an appropriate mean-difference interval can target the geometric mean of paired speed ratios; it does **not** automatically estimate a ratio of medians or a ratio of arithmetic means. Do not divide separately averaged mismatched statistics or rename the resulting quantity after observing it. The analysis must also represent the actual hardware/session and training-run dependence; the existing unit-seed interface is not automatically suitable for every timing design.

These endpoint requirements remain implementation-admission gaps until the campaign demonstrates that its chosen estimator matches each frozen scientific quantity.

## 4. Sample size and power

**Five independent training seeds per admitted configuration is an initial planning default, not a power guarantee or a universal minimum.** A second meaningful model/data scale is a planned confirmation requirement for broad architecture claims; it does not substitute for independent seeds at either scale. Select scales and training lengths using the claim, baseline competence, learning curves, and resource feasibility.

Use a separate development/pilot stage to estimate training-seed variance, family variance, candidate-baseline covariance, discordant binary outcomes, and runtime variability. Then simulate power and interval coverage under the intended crossed design and multiplicity correction. Plan for the smallest improvement worth adopting and the proposed noninferiority margins, using a predeclared target such as 80–90% power if suitable. Those percentages are planning targets, not achieved properties of the present campaign.

Increase seed count when between-training-run variation dominates; increase genuinely independent families when task variation dominates. More paraphrases from existing families do not produce equivalent additional information. Select the confirmatory seed and unit counts before unblinding. If the available budget cannot support a decisive design, narrow the prospective claim or report an inconclusive comparison. Do not reinterpret a wide interval as evidence of equivalence.

## 5. Quality and efficiency decision rules

Predefine a small primary endpoint set and the workloads on which quality must be preserved. Every numerical margin in the campaign is a **proposal** until justified and frozen using operational error tolerance, development evidence, and the intended use. Do not derive acceptable degradation from the candidate's final result or change a margin after seeing confirmatory outcomes.

For a higher-is-better quality endpoint, let `delta = candidate - baseline`. Noninferiority requires its appropriately adjusted lower confidence bound to exceed `-quality_margin`. For lower-is-better NLL, noninferiority requires the adjusted upper bound of the candidate-minus-baseline difference to be below the proposed NLL margin. Failure to show a significant difference is not noninferiority.

An efficiency-with-preserved-quality claim requires **both**:

- Every designated primary quality/safety gate passes its noninferiority rule, with no disqualifying feasibility or failure-rate gate violation.
- The adjusted lower confidence bound of the relevant improvement ratio, such as `baseline_latency / candidate_latency`, exceeds the predeclared practically useful threshold.

For a superiority claim, the favorable adjusted confidence bound must lie strictly beyond the predeclared superiority threshold: the lower bound exceeds the positive margin for higher-is-better quality, or the upper bound is below the negative margin for lower-is-better quality. Merely crossing the threshold is insufficient; passing noninferiority alone is insufficient. For a joint better-quality-and-lower-cost claim, both superiority requirements must pass. State whether the claim concerns latency, throughput, memory, energy, or amortized cost; superiority on one does not imply superiority on the others. A global average must not conceal failure on a designated critical workload.

Keep the complete primary comparison family fixed. Per-family computation can allocate alpha with nonnegative allocations summing to the overall budget, but each subanalysis must retain its allocated correction. Selection of the strongest baseline must occur on development data or be accounted for in the confirmatory comparison family. Post hoc explorations receive an exploratory label.

## 6. Failures, timeouts, and divergence

**Evaluation failures:** Keep every scheduled task in the denominator of completed-task or success-within-budget metrics. A crash, invalid response, exhausted budget, or timeout is a failed task for those endpoints. Record the reason and consumed resources. Never report success-only accuracy or remove hard cases because one architecture failed.

**Timeouts and resource exhaustion:** Completion latency is right-censored or infeasible, not zero and not an observed completion at the timeout limit. Report success-within-budget and timeout/OOM rates alongside latency. A mean or median conditioned on successful requests is secondary and explicitly labeled. Capped execution cost can be reported as cost actually consumed before termination; it is not the cost of a completed answer. Do not convert an undefined NLL after a failed run into an invented finite score; mark it unavailable and fail the relevant feasibility/admission gate.

**Training divergence:** Freeze the treatment of failed training runs before confirmatory execution. Architecture-related divergence and budget exhaustion count toward training reliability and research cost; do not replace them with favorable seeds until a desired number of successes appears. A successful-run-only estimate is conditional and cannot support an unconditional reliability claim. Predefine a minimum valid-run requirement consistent with the power plan; missing runs can make the primary comparison inconclusive even if remaining averages look strong.

**External infrastructure faults:** Permit a documented same-configuration, same-seed rerun under a symmetric, predeclared rule for identifiable external faults. Retain the failed attempt, classification evidence, rerun linkage, and actual consumed resources. Do not classify model-induced numerical failure as infrastructure failure without evidence.

**Invalid baselines before testing:** If a baseline fails implementation or training-adequacy gates, suspend the comparison and repair/requalify it under the equal tuning-budget policy. Its invalidity is not evidence that the candidate wins. Any change after test exposure requires preserving the old record and qualifying a fresh confirmatory evaluation; the exposed test set becomes development evidence for that change.

## 7. Required reviewer record

The campaign manifest should preserve: endpoint definitions; unit/family identifiers; seed pairing; all scheduled and completed runs; failure classifications; proposed and frozen margins; tuning and training allocations; power assumptions; multiplicity/alpha allocation; bootstrap convention and validated precision; hardware measurement blocks; claim scope; and a clear admitted/unadmitted decision with reasons.

No result should be promoted solely because `freeze_design` accepted it. The inspected analyzer can emit `status = TESTED` while explicitly retaining `scientific_qualification = NOT_QUALIFIED`; that distinction must survive downstream reporting. Statistical adequacy, training adequacy, implementation fidelity, and workload relevance are separate gates.
