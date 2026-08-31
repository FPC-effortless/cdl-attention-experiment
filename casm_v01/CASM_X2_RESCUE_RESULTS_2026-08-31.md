# CASM-X2 hidden-only convergence rescue — 2026-08-31

## Purpose

CASM-X2 v0 produced a 100% explicit-transition result while parameter-matched hidden-only GRU and Transformer controls underfit even IID depth 4. This rescue deliberately gave only the generic controls substantially more favorable optimization to test whether the original separation was merely a short-run training artifact.

The rescue is **not** the final causal control because a later post-run audit identified a separate teacher-forced previous-state-access asymmetry. CASM-X2A addresses that issue directly.

## Provenance

Workflow run: `33363140352`.

Exact evaluated head: `19cbc47d3228f60c8137cf2ce29932c226130714`.

Seeds:

- `20260851` / eval `20260931`
- `20260852` / eval `20260932`
- `20260853` / eval `20260933`

Parameter counts remained matched to the 246,160-parameter explicit reference:

- GRU: 245,848
- Transformer: 242,852

## One-sided curriculum

Only the generic controls received the extra compute:

- depth 1: 1,500 steps
- depth 2: 2,000 steps
- depth 3: 2,500 steps
- depth 4: 4,000 steps

Total: **10,000 optimization steps per control**, versus 2,400 steps in the original X2 run.

## IID checkpoint trajectory

Mean depth-4 IID final-state exactness across three seeds during the final curriculum stage:

| Total step | GRU | Transformer |
|---:|---:|---:|
| 6,500 | 2.08% | 12.63% |
| 7,000 | 2.60% | 15.10% |
| 7,500 | 1.56% | 15.10% |
| 8,000 | 2.60% | 16.54% |
| 8,500 | 2.47% | 16.54% |
| 9,000 | 3.65% | 16.41% |
| 9,500 | 2.08% | 17.97% |
| 10,000 | 3.39% | 13.93% |

There is some Transformer improvement, but neither model shows convergence toward the preregistered 80% IID competence threshold.

## Final evaluation

Mean exact final-state accuracy across three rescue seeds:

| Suite | GRU | Transformer |
|---|---:|---:|
| IID depth 4 | **3.04%** | **18.84%** |
| held-out composition depth 6 | 1.13% | 0.95% |
| held-out composition depth 12 | 0.43% | 0.09% |
| stress depth 24 | 0.61% | 0.09% |
| stress depth 48 | 0.61% | 0.17% |
| stress depth 96 | 0.43% | **0.00%** |

At depth 96, mean step-state exactness was 0.87% for GRU and 1.88% for Transformer. Mean per-register accuracy was 16.34% and 14.95%, respectively.

## Interpretation

The one-sided rescue fails to rescue either hidden-only control to IID competence.

This rules out the narrow explanation that the original 2,400-step gap was simply caused by a modest shortage of optimization steps. The Transformer benefits from substantially more training on IID depth 4, rising from roughly 6.9% in X2 v0 to 18.8%, but that gain does not transfer into held-out composition or long execution. The GRU remains very weak throughout.

However, the rescue does **not** establish that hidden-state sequence models cannot represent the algorithm. The post-run audit found that the explicit transition model is teacher-forced with the true previous state during training, while these controls are not. More compute does not equalize that information/decomposition advantage.

The justified rescue conclusion is therefore:

> state-exposed local transition learning is dramatically more sample- and optimization-efficient than the tested hidden-only formulations in this controlled contextual machine.

The stronger question — whether the transition architecture still has an advantage after equalizing previous-state access — is delegated to CASM-X2A.
