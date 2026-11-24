# Comparison readiness — 22 November 2026

**Outcome: a rigorous prospective campaign can be specified; a modern trained head-to-head comparison is not yet executable from the verified assets.** No model benchmark was run in this work.

## Verified starting points

| Item | Observed state | Evidence |
|---|---|---|
| Candidate lineage | nanoCOG packet reference on draft PR88; separate from older COG-LM | [Pinned README](https://github.com/FPC-effortless/cdl-attention-experiment/blob/8374f72ba9324fa4a11d0d3ecb40a6db0e311c73/research/nanocog-001/README.md) |
| Existing scientific contract | Already separates software, mechanism, language and cost; prospective, not admission | [Pinned qualification](https://github.com/FPC-effortless/cdl-attention-experiment/blob/8374f72ba9324fa4a11d0d3ecb40a6db0e311c73/research/nanocog-001/QUALIFICATION.md) |
| Training/serving parity | Tokenizer, initialization, optimizer grouping, batching and serving not drop-in equivalent to nanochat | Candidate README |
| Current packet implementation | Dense training surrogate; Python time loop; functional state copying; owned inference path in source | [Packet specification](https://github.com/FPC-effortless/cdl-attention-experiment/blob/8374f72ba9324fa4a11d0d3ecb40a6db0e311c73/research/nanocog-001/nanocog/comm/README.md) |
| Recent learned diagnostic | Nine 37,414-parameter fits; four TRAIN-adequate; no arm passed all held local-binding gates | [COG-JOINT-LOCAL-001](https://github.com/FPC-effortless/cdl-attention-experiment/blob/8374f72ba9324fa4a11d0d3ecb40a6db0e311c73/research/nanocog-001/cog-joint-local-001/JOINT_RESULT.md) |
| Existing statistics | Paired crossed seed/unit analyzer, equal-cell estimator, bounded computation | [Veritas source](https://github.com/FPC-effortless/veritas/blob/eec439492407b0aa26922c23b5ae40d95289f4eb/integrations/trainlab/src/trainlab/comparison_design.py) |
| Current local runtime | Python 3.12.14 x86_64; Torch module not found; nvidia-smi not found | Direct environment inspection in this task |
| Native modern model runs here | Zero | No model imports, fits, checkpoint loads or timing runs performed |

Absence of nvidia-smi is a local visibility finding, not proof that the user has no accessible accelerator elsewhere. No remote accelerator allocation or paid runner was inspected or established.

## Work that must precede a scientific head-to-head

1. Bind a selected candidate graph/configuration and common tokenizer, data pipeline, optimizer parameter groups and complete state adapter. Do not relabel the older COG-LM model as the current candidate.
2. Integrate the core GQA, MLA and explicitly named KDA+GQA comparisons, preserving meaningful common-backbone and separate capacity views.
3. Verify full/chunk/step equivalence, causality, reset/fork isolation, gradients, actual state allocation and true backend selection.
4. Profile admitted TRAIN/CAL runs on a known runner; choose model size and training length that demonstrate baseline competence.
5. Validate the statistical implementation for token-weighted NLL, paired cost ratios and the intended corrected interval precision. Existing numerical input acceptance is insufficient.
6. Freeze exact resource caps, primary cohort, margins, final roster, seeds and sample-size rationale, then obtain independent exact-version stage review before fits/prediction/scoring.

Source inspection and planning did not fail because the user withheld permission. Execution prerequisites and actual runner capacity remain unresolved. The user has requested the comparison; the proposal does not invent a budget, data rights, checkpoint or successful admission.

## CSA2 limitation

The reviewed [NVIDIA NeMo implementation](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/deepseek_v41/attention) freezes released indexer weights because hard top-k has no language-model gradient and the source release lacks indexer distillation. Its training implementation also documents cached decode as unsupported. This is a constraint of the inspected implementation, not a claim that no CSA2 implementation can decode. A separate complete training/inference integration could qualify it later.

## Reporting boundary

All modern comparators remain PENDING or CONDITIONAL. There is no winner and no measured modern-comparison speedup. No protected cohort has been opened; no compute purchase or model job was submitted. A public proposal/index entry records planned work only and preserves historical negative results.

