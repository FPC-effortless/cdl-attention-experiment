# Reproduce the authentic issue-history diagnostic

The source is the GitHub REST API for public issue search, issue timelines and issue state. The freeze covered the source filters, calendar splits, candidate cutoff, model specifications, threshold grid and acceptance gates. `recipe/frozen-plan.sha256` pins the analysis plan and executables. `recipe/source-manifest.json` records the captured issue selectors and raw source digests. Each comment and state transition has its permanent source identifier in the local evidence. The GitHub handoff can be replayed from source, but historical bodies, edits, permissions and search indexes can change after the archived fetch. Exact replay uses the captured local corpus.

Research Lab OS, TrainLab, and the Veritas evidence model came from [Veritas at eec4394](https://github.com/FPC-effortless/veritas/tree/eec439492407b0aa26922c23b5ae40d95289f4eb). The original experiment ran on Linux, Python 3.12, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, ONNX Runtime 1.24.4 and Transformers 4.51.3. The pinned quantized [DeBERTa NLI checkpoint](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) and tokenizer were reused byte-for-byte from the [semantic-transfer evidence bundle](../semantic-transfer/README.md) in `recipe/nli/`. The checkpoint is upstream Apache-2.0. Raw GitHub comment redistribution rights were not established.

For a fresh collection, run the following in an authenticated Python environment with the Veritas integrations on the import path. `acquire.py` uses `GH_TOKEN` or `GITHUB_TOKEN` for public reads. A complete captured `data/` directory skips the first command. The endpoint cross-check re-requests only the discrepant issue records and is supplementary.

```bash
sha256sum -c recipe/frozen-plan.sha256
python acquire.py
python prepare.py
python check_snapshots.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python launch.py semantic
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python launch.py
OPENBLAS_NUM_THREADS=1 python audit.py
```

The run directories contain the validation-selected configuration, coefficients, test probabilities, admissions and artifact hashes for each arm and seed. Saved `joblib` files should only be loaded from a trusted source. The audit recomputes source labels, counts, metrics and replay digests from the evidence. The three grouped cross-fitting seeds use one corpus. The clean-directory replay verifies reproducibility on the same host.
