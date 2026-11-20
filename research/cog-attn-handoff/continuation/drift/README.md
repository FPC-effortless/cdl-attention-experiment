# Confirmed command reuse under changing authority

Research Lab OS froze the protocol and TrainLab ran six jobs using the same three checkpoint/trace pairs as WR-010. Three write probabilities (10%, 50%, 90%) produce 6,921 writes and 6,903 reads per mode. Each learned proposal on a cache miss is staged and confirmed by a separate local process, which returns the authoritative action and the dependencies it read. Exact-command hits may skip that trip.

In each 32-write block the experiment changes the selected provider's operation mapping, revises an inactive provider, switches the authority selector, then changes an active mapping between lookup and commit. Each event targets the current command's group. Provider and selector epochs have synchronous local visibility. This deliberately concentrates races on active writes. Changes can flip copy versus addition; some authority changes preserve the resulting action.

| Guard | Wrong committed actions | Reads below full reward | Confirmation calls |
| --- | ---: | ---: | ---: |
| Reuse exact string | 2,990 | 2,351 | 1,137 |
| Trace provider, omit selector | 767 | 960 | 3,731 |
| Full trace checked before use only | 213 | 772 | 4,689 |
| Full trace checked again at commit | 0 | 0 | 4,820 |
| Global generation with commit check | 0 | 0 | 6,253 |
| Always confirm and retry on a raced write | 0 | 0 | 7,134 |

The full trace saved 1,433 confirmations (22.9%) relative to the global epoch. The precheck-only guard lost every one of the 213 injected races. The independent replay derives action meanings from a pair of provider counters and a selector per group, then recomputes intended state. Veritas CompanyWorld scored 41,418 saved read responses, and six TrainLab receipts passed integrity checks. The Forge-backed ecosystem comparison also passed reverification.

This experiment adapts the selective invalidation and traced-branch principles documented in the 042c and 042e handoff reports. The new implementation has a synchronous event stream and a serialized final check and state update. It does not establish a distributed atomic commit protocol, discover silent drift, or show that historical state remains correct after a retroactive reinterpretation. The verifier is truthful. The rates are for this constructed fault schedule and are unsuitable as estimates of real-world incidence.

References: [042c lifecycle report](https://github.com/FPC-effortless/cdl-attention-experiment/blob/research/cog-attn-reproducibility-20261120/research/cog-attn-handoff/lifecycle/README.md) and [042e lifecycle report](https://github.com/FPC-effortless/cdl-attention-experiment/blob/research/cog-attn-reproducibility-20261120/research/cog-attn-handoff/lifecycle-v2/README.md).

## Memory scope

Summing uncompressed NumPy parameter-array bytes in each of the three preceding latency comparisons gives 385,968 bytes for WR and 463,304 for the matched decoder. These figures exclude process libraries, allocator overhead, activations, the KV cache, the command and response caches and the verifier. Peak or steady-state working-set superiority was not measured.

## Reproduction

Use Python 3.12, NumPy 2.3.5 and Veritas, Forge, TrainLab and Research Lab OS at eec439492407b0aa26922c23b5ae40d95289f4eb. From a fresh directory with the recipe inputs and launch and audit scripts, run:

    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python audit.py
    veritas-ecosystem verify "$PWD/ecosystem"

The archive contains the frozen recipe, checkpoints, per-write commitments and call counts, saved read submissions, audit rows and evidence database. No new model fits or independent scientific review were performed.
