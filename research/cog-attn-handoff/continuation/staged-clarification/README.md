# Staged clarification under noisy and dependent replies

This cycle tests the residual error after single-slot clarification. It uses the prior frozen separate-role DeBERTa NLI encoder and a fresh template bank across seeds 751, 773 and 797. The four verb bindings in each seed come from 96 support transitions. The earlier clear calibration bank fits the posterior cutoff before evaluation. Each regime has 768 determinate commands and 384 indeterminate commands. All policies receive paired synthetic reply tapes and the same frozen action probabilities.

The contextual model scores language. A policy decides whether to release a typed write. The experimental quorum requires two current matching evidence roots for each argument and at least .99 operator probability. It checks that source and destination differ and defers on an explicit unresolved answer from a current source. For comparison, the service-label quorum treats different service labels as different roots. The outcome audit distinguishes wrong actions from wrong effects, since a wrong action can coincidentally yield the intended state.

## Main results

| Reply profile and policy | Determinate admissions | Wrong actions | Wrong effects | Indeterminate admissions | Mean field requests per determinate command |
| --- | ---: | ---: | ---: | ---: | ---: |
| Truthful, one response | 768 | 101 | 85 | 0 | 1.00 |
| Truthful, one response per argument | 768 | 0 | 0 | 0 | 2.00 |
| Independent noise, one response per argument | 597 | 29 | 20 | 1 | 1.90 |
| Independent noise, provenance quorum | 758 | 0 | 0 | 0 | 5.09 |
| Shared snapshot, service-label quorum | 753 | 127 | 119 | 51 | 5.02 |
| Shared snapshot, provenance quorum | 705 | 3 | 2 | 0 | 5.29 |
| Shared snapshot, calibrated provenance posterior | 441 | 0 | 0 | 0 | 6.40 |
| Concealed shared snapshot, provenance quorum | 753 | 127 | 119 | 51 | 5.02 |

With genuinely independent errors, the provenance quorum passed the gate at 98.7% coverage and zero observed wrong commitments. With visible dependency it committed 702 correct and 3 wrong actions, giving 91.8% determinate coverage and 99.57% precision. The posterior cutoff chosen on clean calibration was .9999. It had zero wrong admissions on the visible shared condition and missed the coverage gate at 57.4%. Counting separate names as separate evidence failed when their replies came from a common stale meaning. Hiding that common source erased the provenance-aware policy's gain.

## Separately acquired reserve

The follow-up freezes a policy that queries only services 4–6. The experiment supplies their acquisition independence and keeps the held-out commands while drawing fresh error tapes. Both policy variants remain fixed across the follow-up profiles.

| Follow-up profile and policy | Determinate admissions | Wrong actions | Wrong effects | Indeterminate admissions | Mean requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Concealed primary dependency, public quorum | 750 | 102 | 98 | 48 | 5.09 |
| Concealed primary dependency, reserve quorum | 593 | 1 | 0 | 0 | 4.76 |
| Reserve unavailable, reserve quorum | 0 | 0 | 0 | 0 | 6.00 |
| Shared fault reaches reserve, reserve quorum | 601 | 92 | 90 | 45 | 4.75 |

The separate reserve met the three empirical gates at 77.2% coverage, 99.83% precision and zero indeterminate admissions in the hidden-primary profile. Zero effect errors on the admitted commands reflects one incorrect action with the same numerical effect. During the outage it deferred all commands. Corrupting both channels again produced incorrect and indeterminate commits. The system makes no claim that source independence can be recovered from names, cryptographic identity, confidence or voting alone.

## Scope, audit and replay

This is typed offline clarification. It uses simulated availability, exact or substituted slot values and supplied epoch/root metadata. Request cost counts fields even when a reply is stale or missing. Real response latency, human reliability, cross-host consistency and acquiring genuinely independent evidence were outside the fixture. The upper bound consumes three truthful typed fields and only releases determinate commands. A finite zero-error count is an observation on these fixtures.

Research Lab OS registered 12 TrainLab jobs. The audit checked receipts, regenerated input labels and reply tapes, verified the structural support for quorum commits, and separately replayed 39,168 policy outcomes. The enclosed Veritas ecosystem reference comparison passed reverification. The policy experiments use the independent domain arithmetic oracle.

Use Python 3.12, PyTorch 2.15 CPU, NumPy 2.3.5, transformers 4.51.3, ONNX Runtime 1.24.4 and the Veritas ecosystem at eec439492407b0aa26922c23b5ae40d95289f4eb. Execute from a fresh directory:

    (cd recipe/nli && sha256sum -c ../../nli.sha256)
    export TOKENIZERS_PARALLELISM=false
    veritas-ecosystem run "$PWD/ecosystem"
    python recipe/fresh_scores.py > generation.log
    python launch.py
    python launch_reserve.py
    python audit.py > audit.log
    veritas-ecosystem verify "$PWD/ecosystem"

The model is the upstream quantized AVX2 DeBERTa artifact at revision a150876415327c80daeff35ca6f68f5ed8cf5c24. The bundle includes exact model bytes, the precomputed new scores, the 384 calibration examples drawn from the previous NLI experiment, the frozen recipes, recorded replies and audits. The score checksums bind the exact policy inputs even if a future inference stack changes float-level scores.
