# CASM-Bench v1 — experiment scorecard

Use this order for every architecture interaction. Do not omit a field because it is unfavorable.

## Identity
- benchmark: CASM-Bench-v1.0
- model/variant:
- checkpoint SHA256 / Git commit:
- training seed:
- training steps/tokens:
- training-domain declaration:
- DEV-CORE digest: `a08e78746d3f3f93e05137f701fd7f6c734c5437beccf5ed0d0a2c1dacda5a0a`
- DEV-OOD digest: `67880ecad3fe300f724a7e27e7b3c873aae44dd09e526b12a998cba0a8fbf52e`

## Primary capability
| suite | NormalizedSolveMacro | RawSolveMacro |
|---|---:|---:|
| DEV-CORE | | |
| DEV-OOD | | |

## Per-task free-generation exact accuracy
| task | CORE | OOD |
|---|---:|---:|
| associative retrieval | | |
| state tracking | | |
| arithmetic | | |
| rule induction | | |
| graph reachability | | |
| reverse/copy | | |

## Collapse / baseline diagnostics
- graph CORE: yes accuracy / no accuracy:
- graph OOD: yes accuracy / no accuracy:
- tasks at or below majority baseline:
- missing/unterminated generations:

## Probability diagnostics — not solve metrics
- teacher-forced answer NLL by task, CORE:
- teacher-forced answer NLL by task, OOD:
- answer-byte accuracy TF:

## Efficiency
- parameters:
- recurrent reasoning steps:
- DEV-CORE evaluation seconds / cases per second:
- DEV-OOD evaluation seconds / cases per second:
- training wall time:

## Causal mechanism checks
- memory zero/shuffle ablation:
- recurrent-depth 1/3/5 sweep, if applicable:
- other architecture-specific falsifier:

## Contamination
- exact DEV prompt overlap with recorded training hashes:
- HOLDOUT accessed? yes/no
- if HOLDOUT: `certified_clean` and overlap count:

## Decision
Allowed labels:
- `REJECTED` — primary capability does not improve or falsifier fails.
- `DIRECTIONAL` — one-seed DEV improvement; replicate before promotion.
- `REPLICATED` — >=3 seeds satisfy DEV promotion rules.
- `PROMOTED` — replicated candidate also passes contamination-certified HOLDOUT.

Decision:
Reason:
Known regressions/tradeoffs:
Next falsifier:
