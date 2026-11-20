# WR-005 residual mass follow-up

Three fresh seeds at 32 entities. Both inverse-inclusion weighted arms passed every predeclared accuracy, changed-transition shortlist recall and check-budget gate in all three seeds. See [protocol.md](protocol.md) for criteria and estimator.

| Training | Familiar | Held pairs | Recombined | Checks/example | Seconds/seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exhaustive | 98.47% | 94.50% | 97.92% | 2,048 | 5.30 |
| Destination-conditioned fallback | 57.29% | 51.17% | 54.23% | 10.35 | 0.79 |
| Always 16 residual, unit weights | 81.84% | 73.89% | 80.27% | 24 | 1.30 |
| Always 16 residual, inverse inclusion | 97.20% | 92.42% | 96.97% | 24 | 1.25 |
| Always 32 residual, inverse inclusion | 97.79% | 93.36% | 97.59% | 40 | 1.30 |

The 16-draw weighted arm used 85.3× fewer compatibility checks and 4.2× less training-loop time than exhaustive training. All runs use the same initialization within each seed. The executor has a single write, which makes the destination restriction exact for changed transitions. The synthetic model uses n-gram text features. TAC/LRC integration remains open.

## Retained mass and gradient alignment

Held pairs at the original destination-conditioned checkpoints, pooled over three seeds and five residual draws per example. The deterministic top-8 is measured once. Retained mass measures the unweighted compatible mass physically selected. Estimated mass incorporates any weights. Both are ratios to the exact compatible mass. Mean per-example logit-gradient cosine excludes zero gradients. Relative norm error includes skipped examples.

| Transition | Proposal | Support | Retained mass | Estimated mass | Gradient cosine | Relative norm error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Changed | Top-8 | .920 | .854 | .854 | .987 | .186 |
| Changed | Fallback | 1.000 | .897 | .897 | .976 | .146 |
| Changed | Always 16, unit | .999 | .894 | .894 | .968 | .143 |
| Changed | Always 16, weighted | .999 | .894 | 1.003 | .964 | .167 |
| Changed | Always 32, weighted | 1.000 | .930 | .994 | .981 | .105 |
| No-op | Top-8 | .967 | .801 | .801 | .892 | .969 |
| No-op | Fallback | 1.000 | .802 | .802 | .877 | .994 |
| No-op | Always 16, unit | 1.000 | .803 | .803 | .876 | .994 |
| No-op | Always 16, weighted | 1.000 | .803 | 1.027 | .818 | 1.635 |
| No-op | Always 32, weighted | 1.000 | .804 | .976 | .811 | 1.503 |

The paired 16-draw arms have the same support and retained mass at fixed parameters. The weighted estimate is approximately centered on the exact mass in this slice. It still has poorer local gradient alignment on no-ops. The training result isolates weighting at a fixed proposal budget but does not establish that better checkpoint-level gradient cosine mediated the gain. Measuring gradient behavior during training and reducing no-op sampling variance remain open.

## Replay

Python 3.12.14, NumPy 2.3.5, Linux x86-64, single-thread OpenBLAS. Verify the shipped files with sha256sum -c SHA256SUMS. Run in a copy to preserve timing records:

    python diagnose.py
    python run.py > training.log
    python audit.py > audit.log

The diagnostic checkpoint inputs are unchanged copies from the original pilot. Results preserve per-seed scores, check counts and timings. The audit replays held-out model predictions through explicit state mutation and dense candidate ranking and checks the weighted-arm gates. Its generator, tokenizer and forward pass are shared with training.
