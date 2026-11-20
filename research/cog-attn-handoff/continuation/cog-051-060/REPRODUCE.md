# Reproduction and evidence access

## Verify the public handoff

Run from this directory in a fresh checkout of the research branch. Python 3.12 is sufficient for the manifest and numeric audit. The physical re-induction requires NumPy and scikit-learn. The hosted job pins both. public_data_replay.py independently re-fetches the pinned EPIC and MultiWOZ sources, fits both models and checks the published operating points. Running from the committed predictions avoids live provider drift.

```bash
sha256sum --quiet -c SHA256SUMS
python audit.py
python audit_transfer.py
python external_replay.py
```

The two audits check 45 issue ranking tables, 960 session traces, 7,680 primary scaling answers, 19,336 kitchen predictions, 29,488 dialogue decisions, 56 delayed commits, 23,040 supplemental scaling operations and the 12 controlled learner checkpoints. Counts from different experimental arms are repeated evaluations of the same cases. The record in veritas-evidence.json pins the manifest. It records one producer's observed evidence. A hosted CPU reran the specified induction and primary audit from a separate environment. [Workflow](../../../../.github/workflows/cog-acquisition-replay.yml) · [Run receipt](results/github-replay.json)

## Rebuild the study

Use a fresh writable checkout for a rebuild. The scripts write results alongside the source. Preserve the committed outputs for comparison. Timings vary. The reference environment used Linux, Python 3.12, CPU PyTorch 2.15.0, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, ONNX Runtime 1.24.4, Transformers 4.51.3 and Pydantic 2.13.4. Node.js runs the snapshot guard. Export the following from an environment containing those dependencies.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VROOT=/workspace/scratch/veritas
export PYTHONPATH="$VROOT/src:$VROOT/integrations/research_lab/src:$VROOT/integrations/trainlab/src${PYTHONPATH:+:$PYTHONPATH}"
export COG_PRIOR_ROOT=/workspace/delta-authority-048
export COG_RUN_DIR=/workspace/cog-next-private
mkdir -p "$COG_RUN_DIR/epic"
```

Check out FPC-effortless/veritas at eec439492407b0aa26922c23b5ae40d95289f4eb. Extract the published delta-authority-048 evidence so its recipe, portable heads and original run directories are directly inside COG_PRIOR_ROOT. Use the continuation downloader for the pinned evidence revision. prior-inputs.json lists the expected SHA-256 of each actual bridge dependency. [Veritas](https://github.com/FPC-effortless/veritas/tree/eec439492407b0aa26922c23b5ae40d95289f4eb) · [Original module evidence and downloader](../README.md) · [Prior input pins](prior-inputs.json)

### Controlled panels, actual modules and memory

```bash
python launch.py
python discovery_skills.py
python grammar_repair.py
python feedback_lm.py
python policy_repair.py
python reuse_profile.py
python continual_repair.py
python joint_runtime.py
python online_cost.py
python parallel_cpu.py
```

launch.py uses the Veritas controller and TrainLab worker, registers candidate/control jobs, verifies run records and checks same-host replays. The readout and session supplements use the resulting primary runs. joint_runtime.py reads the original quantized NLI weights and tokenizer from /workspace/issue-history-study/recipe/nli; install that directory before running the integrated supplement. For the 100-task comparison, place the two upstream source files under the run directory and verify their digests before import.

```bash
curl -fsSL https://raw.githubusercontent.com/pathwaycom/bdh/400e1d23a1f3680297383c2f58b3ec420fa3a2fe/bdh.py -o "$COG_RUN_DIR/bdh.py"
curl -fsSL https://raw.githubusercontent.com/DavidVivancos/Neuraxon/fbbcdcd7fa74d1f84ca1a8bc64060441e94d66a3/neuraxon.py -o "$COG_RUN_DIR/neuraxon.py"
(cd "$COG_RUN_DIR" && sha256sum -c - <<'PINS'
cfe24008f920965cc3c8236feff52c89ca794a31e52324acf9ddb4cd6fd50ac9  bdh.py
bee47ac6463db65a00471b6b3f977e3a90769242d1a54038f6a75950bd5ba65b  neuraxon.py
PINS
)
python skills100.py
```

### Physical language and dialogue

Download the EPIC-KITCHENS-100 official train and validation annotation CSVs from the pinned revision. The kitchen scripts expect the exact workspace path shown above. kitchen.py fits and saves the shareable serialized kitchen model to the input directory. The public copy is under checkpoints/kitchen. Dialogue model weights and vectorizers are under checkpoints/dialogue. Load serialized Python models only from a trusted checkout.

```bash
for split in train validation; do
  curl -fsSL "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/EPIC_100_${split}.csv" -o "$COG_RUN_DIR/epic/EPIC_100_${split}.csv"
done
python kitchen.py
python seeded_kitchen.py
python latent_kitchen.py
python kitchen_calibration.py
export MULTIWOZ_DIR=/workspace/scratch/multiwoz/data/MultiWOZ_2.2
python dialogue_transfer.py
python dialogue_session.py
```

For MultiWOZ check out budzianowski/multiwoz at fe0c8e65cfcd8462bd33c86e35f21addc84ca82b. The program reads the distributed train, dev and test shards, then records their digests. Only observation IDs, predictions, state labels and scores are in the public prediction tables. A second dialogue fit reproduced both result files byte-for-byte. EPIC publication uses the source's noncommercial license and attribution. MultiWOZ credits Paweł Budzianowski and contributors. [EPIC license](EPIC-LICENSE.txt) · [MultiWOZ license](MULTIWOZ-LICENSE.txt)

### Public issue corpus

The fresh TrainLab models train and calibrate on the earlier study's archived training and validation rows. Reproduce or restore that corpus first under /workspace/issue-history-study, including recipe/dataset.json, recipe/semantic.npy and recipe/nli. Its public directory contains source and selectors; the captured raw text and learned vocabulary are restricted locally. [Earlier reproduction guide](../issue-history-study/REPRODUCE.md)

Copy fresh-source into /workspace/issue-history-followup, with its recipe subdirectory. A read credential in GH_TOKEN or GITHUB_TOKEN is required for the original public GitHub collection rate. The selector covers July–September 2026. The source corpus was captured on November 20, 2026. The collector resumes from previously fetched pages. On an existing original capture skip reacquisition. Source digests distinguish a faithful replay from a later resampling.

```bash
mkdir -p /workspace/issue-history-followup
cp -R fresh-source/. /workspace/issue-history-followup/
cp -R /workspace/issue-history-study/recipe/nli /workspace/issue-history-followup/recipe/
(cd /workspace/issue-history-followup && python acquire.py && python prepare.py && python recipe/semantic.py)
python run_authentic.py
python replay_authentic.py
python matched.py
node snapshot.mjs
python chronology.py
python confirm.py
python new_snapshot.py
python uncertainty.py
python authentic_session.py
python summarize.py
```

Live endpoint checks can diverge after edits, late events, or repository changes. Training uses observed prefixes. Only the delayed-observation arm is allowed to see subsequent confirmations. The archived source digests, public issue identifiers, numeric per-comment predictions, run receipts and verification waits identify the evidence used. Replay of raw text from the original bytes also requires the restricted captures. [Fresh source manifest](results/fresh-source-manifest.json) · [Recorded verifications](results/confirmation.json)

archive_search.py searches the available branch-head trees and release names for the ten named historical archives. certify.py creates a new manifest and timestamped Veritas record for a rebuilt copy. Preserve the original Git revision and compare the stable outputs separately from timing and mutable provider data.
