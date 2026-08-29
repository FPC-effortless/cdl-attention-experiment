# Stage A paired SmolLM2 results — 2026-08-29

## Provenance

- Model: `HuggingFaceTB/SmolLM2-135M`
- GitHub Actions run: `33231291796`
- Reviewed head: `2ee0fac8886b60c0fd9737d7ec09fedc1770ebe9`
- Device: GitHub-hosted CPU runner
- Benchmark seed: `20260829`
- Cases: 120 paired underlying cases per candidate-count condition
- Candidate counts: 6, 12, 24
- Actions artifact: `9708661555` (`stage-a-results`)
- Artifact digest: `sha256:3988e63f95a8e859d9015ee9514fdfa40583647bfbe300288d2f7b40a1dedef8`

## Retrieval results

| Candidates | Method | Top-1 | Top-3 | MRR | Median rank |
|---:|---|---:|---:|---:|---:|
| 6 | gzip conditional | 0.6333 | 0.9250 | 0.7800 | 1 |
| 6 | SmolLM2 conditional description length | **0.8750** | **1.0000** | **0.9361** | 1 |
| 12 | gzip conditional | 0.5750 | 0.8250 | 0.7139 | 1 |
| 12 | SmolLM2 conditional description length | **0.8250** | **0.9917** | **0.9090** | 1 |
| 24 | gzip conditional | 0.4833 | 0.7000 | 0.6182 | 2 |
| 24 | SmolLM2 conditional description length | **0.7583** | **0.9583** | **0.8669** | 1 |

Top-1 CDL minus gzip:

- 6 candidates: +24.17 percentage points
- 12 candidates: +25.00 percentage points
- 24 candidates: +27.50 percentage points

Approximate Wilson 95% intervals for Top-1:

- 6 candidates — gzip: 54.4–71.4%; CDL: 80.4–92.3%
- 12 candidates — gzip: 48.6–66.0%; CDL: 74.7–88.3%
- 24 candidates — gzip: 39.6–57.2%; CDL: 67.4–82.6%

## Downstream answer NLL

Lower is better.

| Candidates | gzip-selected | CDL-selected | Oracle relevant | Query only |
|---:|---:|---:|---:|---:|
| 6 | 6.5288 | **5.2170** | 4.6415 | 10.5758 |
| 12 | 6.8296 | **5.4006** | 4.6415 | 10.5758 |
| 24 | 7.3734 | **5.7506** | 4.6415 | 10.5758 |

CDL closes approximately 90.3%, 87.2%, and 81.3% of the query-only-to-oracle NLL gap at 6, 12, and 24 candidates respectively.

## Runtime

- 6 candidates: 128.44 s (includes current native-attention diagnostic capped at 80 cases)
- 12 candidates: 114.22 s
- 24 candidates: 194.82 s

## Interpretation

The Stage-A hypothesis survives this synthetic paired test: ranking memories by `-NLL(query | memory)` is materially better than gzip-style conditional compression, and the selected memories improve downstream answer prediction substantially.

This is evidence for conditional description length as a useful relevance signal. It is **not yet evidence that CDL replaces Transformer attention**.

## Important limitations / next falsifiers

1. The benchmark is synthetic relational language; natural-language and long-context tests remain required.
2. The current native-attention diagnostic is not an apples-to-apples retrieval baseline because each candidate memory is evaluated in isolation. Its measured 6-candidate Top-1 (0.1125) must not be used as evidence that CDL beats standard attention. The next version should concatenate all candidate memories into one context and measure query-to-memory-span attention mass.
3. The CDL method is computationally expensive because it performs conditional language-model scoring for every candidate. Stage B must test whether its score can be distilled into a cheap Q/K router without losing the advantage.
4. The benchmark currently reports aggregate correctness. A subsequent run should persist every case's rankings so paired statistical tests and failure clustering are possible.
5. Results must be tested on paraphrases and natural corpora to rule out template-specific exploitation.
