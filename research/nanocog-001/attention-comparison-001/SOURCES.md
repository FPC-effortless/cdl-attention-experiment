# Sources and scope

Repository sources were read through the connected repository service. External mechanism documentation was checked directly. Live upstream references must receive immutable commit/version pins before experiment admission.

## Candidate and governing evidence

- CDL candidate source: https://github.com/FPC-effortless/cdl-attention-experiment/tree/8374f72ba9324fa4a11d0d3ecb40a6db0e311c73/research/nanocog-001
- Candidate draft PR88: https://github.com/FPC-effortless/cdl-attention-experiment/pull/88
- nanochat reference: https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
- Veritas statistical design: https://github.com/FPC-effortless/veritas/blob/eec439492407b0aa26922c23b5ae40d95289f4eb/docs/ecosystem/comparison-design.md
- Veritas analyzer: https://github.com/FPC-effortless/veritas/blob/eec439492407b0aa26922c23b5ae40d95289f4eb/integrations/trainlab/src/trainlab/comparison_design.py
- Lab operating ledger: https://github.com/FPC-effortless/veritas/issues/482
- Current inspected SECANT checkpoint: https://github.com/FPC-effortless/veritas/issues/482#issuecomment-7089186398
- Existing qualification/README/results index/packet specification and JOINT_RESULT.md are linked in READINESS.md.

## Primary external references

- GQA paper: https://arxiv.org/abs/2305.13245
- SDPA documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- FlashAttention implementation: https://github.com/Dao-AILab/flash-attention
- FlashAttention-4 paper: https://arxiv.org/abs/2603.05451
- MLA inference reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
- FLA source: https://github.com/fla-org/flash-linear-attention
- FLA installation and CPU caveat: https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md
- Kimi Linear report: https://arxiv.org/abs/2510.26692
- Gated DeltaNet-2: https://arxiv.org/abs/2605.22791
- GDN2 source: https://github.com/NVlabs/GatedDeltaNet-2
- CSA2 report: https://arxiv.org/abs/2609.19969
- CSA2 full-sequence training limitations: https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/models/deepseek_v41/attention

Published large-model results are background evidence. No published throughput or quality number has been inserted into this campaign as if it were a locally measured comparator result. Proposal margins, workloads, scale and seed planning are research decisions requiring pre-exposure freeze, not facts asserted by those papers.

