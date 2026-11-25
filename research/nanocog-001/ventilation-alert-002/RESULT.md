# VENTILATION-ALERT-002: TAC led the learned models; an early-window rule beat them all

TAC-OSM led every learned arm on both drift directions, dropout and compound stress. It did not beat a fixed early-window median rule, which recovered all labels in all six regimes. No model met the prospective advancement gate.

## Matched offline test

Seven trainable arms, three paired seeds (173, 271, 907), 1,024 fresh nominal training histories, 192 AdamW updates at batch 32 per fit and 384 held families rendered in each of six regimes. Every arm saw the same 24 × 4 observed histories and the same draw order within seed. No regime flags entered any prediction. This uses constructed building ventilation severity labels, not field telemetry, robotics controls, or claims about deployed alerting. Repeated fit seeds and regime renderings are not independent new scenario families. Each scenario contains two channels and three ordinal severity labels; label balance makes macro recall equal to overall correct fraction per panel.

| Arm | Nominal | Noise | Dropout | Upward drift | Downward drift | Drift + dropout |
|---|---:|---:|---:|---:|---:|---:|
| Last-row MLP | 100.00% | 97.92% | 37.76% | 52.60% | 56.16% | 43.23% |
| GRU | 100.00% | 100.00% | 72.92% | 71.27% | 74.65% | 65.80% |
| Tied SDPA | 100.00% | 100.00% | 45.83% | 72.83% | 73.61% | 51.74% |
| Tied COG | 100.00% | 100.00% | 46.53% | 70.05% | 68.23% | 48.18% |
| Shared span ×1 | 100.00% | 100.00% | 47.22% | 67.01% | 67.45% | 47.40% |
| Shared span ×4 | 100.00% | 100.00% | 47.14% | 66.67% | 67.53% | 47.48% |
| Native TAC observe adapter | 100.00% | 100.00% | 88.72% | 94.97% | 95.14% | 85.59% |
| Whole-window median rule | 100.00% | 100.00% | 100.00% | 66.93% | 69.27% | 90.62% |
| Early-window median rule | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

TAC beat tied SDPA by 33.85 balanced-recall points under compound stress (descriptive paired bootstrap interval +24.48 to +42.71), but still under-alerted in **102/384 severe-case decisions** across the three paired seeds. The whole-window rule also scored 5.03 points higher in that regime, although its paired interval overlaps zero. Warm batch-one forward-pass medians were approximately 3.39 ms for TAC and 0.26 ms for tied SDPA on one CPU. These are model-only timings without matched quality, not evidence of serving superiority.

The benchmark cleanly shows robustness to late sensor corruption, but the generator assigns severity before the window and leaves a stable early segment. An observable early-window rule can ignore all later evidence without penalty. That is a benchmark limitation, not evidence that the rule will identify new or changing real events. The next discriminating test should pair histories with identical early measurements and opposite late true severity, independently of late nuisance drift; require use of newly observable evidence and retain the early/whole-window controls.

## Execution evidence

The actual native TrainLab job completed all 21 fits in 44.266629 s and accepted the recipe- and input-bound observations artifact: EXECUTED_CRITERIA_MET, NOT_QUALIFIED. Three prior no-fit canaries checked acceptance of a correct receipt and rejection of an incorrect recipe digest and observed count; the original 63-fit failed record was preserved. Data replayed byte-for-byte and a separate endpoint scorer implementation reproduced all 126 fit-by-regime sets of correct and severe-under counts. This confirms the engineering receipt and score, not the scientific generality of the fixture.

DATA 9cf9af715c854183669885a8fdc364ddb416b344b694d6641b3b0805de1a992f; LOCK 6bf4c6fa68824eed5add9895fc1729161e601b0f15c62649e689c4f010f11ee8; native run 3de5a359d23b186017a306aa526c3ba121d83a86a5280a90ebb2339a19cb21e1; score 0deb3deec586c3e81812f3504142021ec3c587813eff4a34c27b6b09d8d54f17; endpoint recount 75f838459042e0744480d33509c6bd54ca8f8cbbd365743e2023d7daadc4287d.
