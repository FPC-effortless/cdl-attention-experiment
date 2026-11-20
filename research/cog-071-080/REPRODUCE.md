# Evidence and reproduction

Use a checkout of this research branch with Python 3.12. The measured local CPU environment used NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, ONNX Runtime 1.24.4 and Transformers 4.51.3. The hosted path needs only the first four packages. Exact tests compare action and authorization decisions. Timings and timing-dependent cache admission may differ across hosts.

## Verify archived evidence

Run these from research/cog-071-080. audit.py rebuilds scenario permissions from the seeds, scores first-pass, provisional, and committed decisions separately and reconstructs the summary. It pairs sessions when estimating differences. replay.py re-executes the three reserved sessions for all twelve arms, re-downloads the pinned public banking corpus and MultiWOZ test shards, and checks the supplementary transfer, language and correction panels against their recorded decisions. The runner reuses the published NLI logits. A clean run took minutes on a GitHub-hosted CPU.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum --quiet -c SHA256SUMS
python audit.py
python replay.py
```

## Rebuild experiments in a disposable checkout

The primary protocol was committed at 8c0aaea, before scoring. The original primary recipe sources are the files at c7b50e8, whose SHA-256 digests are recorded in results/lab.json. The current sources add optional repair modes. The hosted stable digest checks the original decision paths. The Veritas dependencies come from FPC-effortless/veritas at eec439492407b0aa26922c23b5ae40d95289f4eb. The program uses its actual Research Lab OS controller, evidence graph, TrainLab runner and run verification. A new lab directory preserves earlier immutable run nodes. The protocol treats the controlled runs as engineering diagnostics.

```bash
export COG_ADAPTIVE_LAB="$(mktemp -d)"
export PYTHONPATH="/path/to/veritas/src:/path/to/veritas/integrations/trainlab/src:/path/to/veritas/integrations/research_lab/src"
python lab.py
python diagnostics.py
python corrections.py
python transfer.py score
python admission.py
```

experiments.py primary and experiments.py rep are standalone alternatives for the two panels. They write stable decision digests alongside raw subsystem and timing records. The original 066 search and 067 acquisition functions, the 068 gateway, the PR #73 filter and the 064 expiry wrapper are vendored byte-for-byte. source-pins.json maps each file to its exact upstream path and commit. The continuous-dynamics 061/063 predictor, 062 binary reranker, 065 codec and complete 068/069 carrier infrastructure use different tasks. Their original results are background evidence. The present comparisons use the applicable actual modules and disclose transferred workloads. The 070 observational source hold was respected.

## Authentic inputs and checkpoints

For Banking77, authentic.py banking downloads the official train/test CSVs at the pinned PolyAI commit. A SHA-256 hash of normalized training text determines the 20% calibration partition, and the official test remains separate. Each result row contains a text digest, gold label, ordered top five, admission flag and exact-overlap flag. No authors' text is republished here. The two fitted public models include their vectorizers. [Source](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b) · [Attribution license](BANKING-LICENSE.txt)

For MultiWOZ use the original 2.2 corpus at budzianowski/multiwoz revision fe0c8e65cfcd8462bd33c86e35f21addc84ca82b and set MULTIWOZ_DIR to its data/MultiWOZ_2.2 directory. The source digests and frozen model come from the sibling 051–060 cycle. The hosted script verifies the test shard hashes before inference. For real GitHub comments, authentic.py issues uses /workspace/issue-history-study/recipe/dataset.json and /workspace/issue-history-followup/recipe/dataset.json. Their exact byte digests are in results/authentic-issues.json. Re-fetching an edited public issue can change the corpus. The raw text and fitted text vocabularies for that arm remain in the restricted workspace. [Prior collection and reconstruction guide](../cog-attn-handoff/continuation/cog-051-060/REPRODUCE.md)

To rerun the pretrained arm, set COG_NLI_DIR to the archived cross-encoder/nli-deberta-v3-xsmall tokenizer and model_quint8_avx2.onnx from the prior semantic-transfer bundle. Compare the directory hashes to results/teacher.json. transfer.py teacher builds the table, then transfer.py score tests it. The NLI mixture weight was selected on calibration data only. All unique evaluation utterances were batch-scored without using their labels. Semantic-support gains use additional demonstrations, and timing tables record the batch inference cost independently. [Prior NLI source and environment](../cog-attn-handoff/continuation/issue-history-study/REPRODUCE.md)

```bash
python authentic.py banking dialogue issues
python transfer.py teacher
python transfer.py score
export COG_CHECKPOINT_DIR="$(mktemp -d)"
python preserve.py
python audit.py
```

The restricted and public classifiers were re-fitted and checked against all stored supervised predictions. Public checkpoints and sources are SHA-256 pinned. The restricted issue-model hashes are in results/checkpoints.json. New benchmark outputs and timing receipts may differ even when decisions agree. This package contains software, model, data and replay evidence. The labels do not attest to human intent, external authorization or production safety.

## Reading the costs

sessions.json has one record per seed and arm. elapsed_ns covers the complete local session. times apportions timed stage calls and units records charged work. Multiple primitive attempts can occur inside a planning operation. Task records carry ranked candidates, provisional and final choices, status, rollback counts, permission flags and elapsed nanoseconds. Setup precedes individual task timers. Full session wall time is authoritative for end-to-end comparisons. Simulated verification or clarification work carries a tariff; it does not include real human wait time. Aggregate work and nanoseconds have different units. In transfer.json, add teacher_charge_ns to elapsed_ns to obtain the conservative per-session charge when an NLI teacher is used. The measured teacher batch was shared across sessions; the raw batch receipt permits other amortization horizons. Treat equal-quality comparisons against exact primitives separately from comparisons in which an arm performs fewer or different actions.

issue_workflow.py requires the restricted issue classifier created by preserve.py, the two prior archived issue datasets and the published earlier confirmation observation receipts. With COG_CHECKPOINT_DIR set as above, run python issue_workflow.py to rebuild the authentic two-domain replay. It opens only disposable local SQLite databases and consumes recorded verification events. certify.py creates the public SHA-256 manifest and a timestamped Veritas evidence envelope after rebuilding. Its observed-at timestamp and local timing measurements vary.
