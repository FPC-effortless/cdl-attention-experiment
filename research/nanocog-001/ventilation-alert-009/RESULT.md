# VENTILATION-ALERT-009: three model capacities at fixed exposure

**The 5.2M model test reached a numerical-stability failure; scaling quality is still unknown.** The private GPU attempt logged 14 of 18 completed fits, then stopped on the finite-loss assertion during the next 5.2M COG fit (seed 271). No held outcome artifacts were exported. This is not an architecture-quality failure result.

## Frozen setup

Private Kaggle version 1, Tesla T4, CUDA required, Torch 2.11.0+cu130. Prospective synthetic offline indoor-air observation policy test, same 008 decision setup with fresh 24,000 FIT families and 8,000 held families. No physical sensing or control. Three widths (48, 384 and 720), two tied architectures and three seeds. Matched 1,024 optimizer updates, batch 384, same draws within seed and fixed FP16 autocast protocol. User-requested Kaggle external runtime, not native TrainLab qualification.

| Width | COG parameters | SDPA parameters | Logged completions |
|---:|---:|---:|---:|
| 48 | 24,289 | 23,665 | 6/6 |
| 384 | 1,484,545 | 1,479,553 | 6/6 |
| 720 | 5,202,721 | 5,193,361 | 2/6 |

All fourteen completed-fit console records show 1,024 actual optimizer updates. Each record is a runtime diagnostic; these are not held scores. The run logged seed 173 for both width-720 arms, then failed around 549 seconds. The failed seed/arm follows the frozen execution order. The exception detects nonfinite loss history but does not locate first divergence or establish fp16 as the root cause. After terminal ERROR, only the native log was available; neither REPORT.json nor ACT.npz was exported. Consequently no severe-miss, cost, terminal-pair, capacity or comparative quality conclusion can be recovered from this run.

The exported operator profiler counts before failure are not total FLOPs: for the first seed the width-720 COG and SDPA forwards reported 332,974,128 and 333,711,376 profiler-supported FLOPs (one 16-tick sequence), respectively. Wider arms use more compute even at matched exposure. Scope excludes inference latency claims.

**Next learning step:** use a separately registered stability-first protocol, compare both architectures with the same FP32 policy initially, record first nonfinite step/activation/gradient if any, and export each arm atomically after its fit and evaluation. Do not overwrite or silently rerun this failed version.

- Frozen source SHA-256: 20ef6f24481754240546b9a119247a6b992497dac4bcd1829ffc98aaa632a67a
- Native GPU log SHA-256: 3a06fcdee702a3acb84b82f15af2e71abd321b4202dae608d51004c488da5b78
- Failure recount SHA-256: 24f8535622ce08655bedb217811784d73796e24d0d8d4caf4ad38faa2d44051b
