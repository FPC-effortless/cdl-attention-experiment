# Woyengi model research: resume here

Audit date: 17 November 2026. This is a recovery map, not scientific promotion or a promise that this public repository contains every private artifact. Read this before starting another experiment. A different agent must not need the previous conversation to identify the current work and the known gaps.

## Owner direction and constraints

The goal is a useful, efficient language model with persistent state, not an indefinite sequence of tiny diagnostic successes. Proceed toward end-to-end language and memory evaluation with competent simpler controls. Inference has 12× the priority of training; keep quality requirements separate from cost. No paid compute or protected TEST access. Use Forge → TrainLab → Veritas → Research Lab, freeze protocols before outcomes, retain failures, and independently replay and score all required endpoints. Publish authorized source continuously; do not publish credentials, non-redistributable data, private run records or checkpoints without separate explicit disclosure permission. Draft source publication is not merger approval. Required exact-head CI and genuine review must pass; never bypass protections.

## Verified remote source pins

These commits were resolved on the remote during this audit. They are on separate draft branches, not necessarily `main`. Checking out `main` alone does not recover the recent work.

| Study | Source / PR | Immutable commit | Status |
|---|---|---|---|
| BOUNDARY-017 | [PR 62](https://github.com/FPC-effortless/cdl-attention-experiment/pull/62), `research/csm-boundary-017-draft-20261116` | `7608a276dbbe6e3707e024b7de7456cb491f840e` | Completed fits/replay/scoring and timing; source-only draft |
| CONDITIONAL-018 | [PR 63](https://github.com/FPC-effortless/cdl-attention-experiment/pull/63), `research/csm-conditional-018-draft-20261117` | `2eff614af8303f31376049c9aece37324fd7535d` | WITHDRAWN. Gate lacked source identity; data admitted a global-value-presence shortcut. Retain partial attempts, do not use as valid learning results |
| CONDITIONAL-019 | [PR 64](https://github.com/FPC-effortless/cdl-attention-experiment/pull/64), `research/csm-conditional-019-draft-20261117` | `c5b53d973192d3f1842185a4ec51d55c7e958b38` | All 9 fits / 144 segments complete; independent replay and scoring complete; timing/final package being finalized at this snapshot |
| Forge private snapshots | [Veritas PR 577](https://github.com/FPC-effortless/veritas/pull/577) | `2bbeb0af5062b775104bcca1cb3478031bd53c8b` | Separate dependency draft, not assumed merged |

`research/recovery/PUBLIC_SOURCE_MANIFEST.json` records SHA-256 for the public research files at these commits. Commit identity and content hash are separate from evidence qualification.

## What the completed work does and does not establish

017 is a learned clause-local parser coupled to supplied identity lexing, a punctuation guard and an exact typed executor. It is not evidence that native CSM discovered the storage semantics or that the system understands unrestricted language. Three seeds of each clause-local design passed the bounded study gates; the added state-conditioned controller did not establish an advantage over the simpler parser. No new Lean compilation was performed.

019 repairs the source-identification defect in the study design, not native CSM as a whole. It compares local parser + learned predicate, state-conditioned parser + same learned predicate, and local parser + exact predicate. Own predicted operands and state are used at execution. All nine runs failed the predefined overall qualification gate on unfamiliar wording. Familiar wording and unseen wording must be reported separately; new operation structures do not imply unseen-name or natural-language competence. Four familiar one-letter identities, four cyclic payload permutations and exposed constructed DEV limit the claim. Do not use favorable familiar-template scores or isolated gate accuracy to declare a passed study. Final measured cost does not override the failed quality gate.

## Recovery inventory: do not silently invent missing history

| Studies | Availability established in this audit | Remaining recovery obligation |
|---|---|---|
| 008–013 | Original source recovered from owner-authorized Library ZIPs into `research/recovered_source/` (including separate MATH-008 and ROUTING-008) | Private dependency/runtime/data closure still required; archival paths are not portable runners |
| POINTER-014 | Original evidence ZIP available in owner's private Library; source recovered in the research workspace | Original Python source/protocol recovered in `research/recovered_source/`; private dependency closure still required |
| COMPOSITION-015 | Original evidence ZIP available in owner's private Library; source recovered in the research workspace | Original Python source/protocol recovered in `research/recovered_source/`; private dependency closure still required |
| BOUNDARY-016 | Original local workspace unavailable after reconnection; last recorded progress was partial | Not recovered. 017 is a distinct replacement, not a reproduction of 016 |
| 017 | Public source pins above; private full recovery and compact evidence retained | Owner-authorized private package access needed for exact replay |
| 018 | Public source preserved; interrupted private evidence retained | Never resume the withdrawn design for its original scientific claim |
| 019 | Public source, private reviewed inputs/admission and all segment backups retained | Obtain final reviewed bundle when finalized; do not assume prospective README status is the latest outcome |

The recovered source is archival, not yet a portable executable package. 63 Python files were syntax parsed; no training or Lean compilation was rerun in this recovery audit. Original runtime pins and data artifacts remain prerequisites.

Private Library filenames to locate through the owner's authenticated Library tools:
- `CSM_POINTER_014_EVIDENCE.zip`
- `CSM_COMPOSITION_015_EVIDENCE.zip`
- `CSM_BOUNDARY_017_FULL_RECOVERY.zip`
- `CSM_CONDITIONAL_019_REVIEWED_INPUTS.zip`
- `CSM_CONDITIONAL_019_ADMISSION.zip`
- `CSM_CONDITIONAL_019_PRIVATE_<arm>_<seed>_<stop_update>.zip`

Do not infer access from filenames. Ask the owner for authorized transfer if the replacement agent cannot access their Library. Private IDs, download tokens and signed URLs intentionally do not appear here. The final 019 evidence package filename is pending verification; search by study name and inspect its contents, rather than assume it exists. Public GitHub alone does not currently reproduce private evidence.

## Start a fresh agent without damaging the active run

1. Read repository `AGENTS.md` and the relevant source protocol; check latest PR status and any later owner direction. Determine whether another worker is still running. Do not launch a duplicate training process.
2. Fetch the exact source commit into an isolated checkout:
   ```sh
   git clone https://github.com/FPC-effortless/cdl-attention-experiment.git
   cd cdl-attention-experiment
   git fetch origin research/csm-conditional-019-draft-20261117
   git checkout --detach c5b53d973192d3f1842185a4ec51d55c7e958b38
   ```
3. Materialize private artifacts with the current supported Library workflow into this executor. Verify ZIP CRC, manifest file roster, SHA-256 hashes and receipt lineage. Keep originals immutable. Do not assume paths from an earlier computer exist here. Do not load untrusted pickle/checkpoint files before establishing provenance.
4. Restore the full dependency source/package closure and exact recorded runtime. 019 used a CPU Torch environment, with full Python/platform/package/Torch build fingerprint in private `RUNTIME_LOCK.json`. Matching only the version string is insufficient. A different environment requires explicit new runtime qualification; never edit the old lock just to pass.
5. The private recovery root contains `candidate/`, native `runs019/`, `DURABLE_RUN_LEDGER.jsonl`, `STATUS.json`, `operator019.py` and `final_review/`. Validate all review/source/operator/runtime pins. A successful subprocess is not a successful scientific gate.
6. For an interrupted admitted run, select only the latest successful native receipt whose archive has a verified durable byte-for-byte roundtrip. Restore its checkpoint index and generations, optimizer, RNG and sampler state. The operator's `advance` function rejects failures, drift, missing or inconsistent backups. Never infer a checkpoint from a filename or silently skip uncheckpointed work. Preserve failed attempts and their cost.
7. Exact commands for a resumed segment depend on the verified recipe path and authorized local package installation. Inspect the recovered operator CLI before use. `PAUSE_REQUESTED` blocks the next segment without changing an in-flight receipt. Stop on uncertainty rather than weakening admission.
8. For 019, training is already complete: do not retrain to finish reporting. Verify the full 144-segment and nine-endpoint roster with the independent completion auditor, then all nine exact-bound checkpoint replays and saved-record rescoring. Keep replay outputs outside immutable run archives. Timing must run with training stopped and check output parity; failed timing attempts remain visible.
9. Deliver the reviewed report and verified evidence to the owner. Update this index and PR status. Do not merge until applicable checks and genuine review pass.

## Mathematical verification boundary

The earlier MATH-008 executor proof is distinct from the learned parser. `MATH.md` in 019 gives the conditional update contract and tests, not a new kernel-checked Lean theorem. Record the actual Lean toolchain, theorem list and `#print axioms` output if a future study compiles new laws. Differential Python tests, abstract proofs, learned selection and end-to-end language performance are four different claims.

## Next research handoff

The owner explicitly asked on 17 November to move quickly toward the end goal rather than continue serial tiny tests. The next experiment should directly assess useful language learning and state use with adequate model/data/training budgets, realistic held-out wording/source groups, meaningful delays and matched local endings, strong simpler controls, full inference costs, and durable recovery from the start. Reuse existing model/data infrastructure where justified. Do not claim that a newly proposed campaign is running or approved merely because this handoff recommends it.
