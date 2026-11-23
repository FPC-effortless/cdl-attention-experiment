# Interactive MoLab engineering notebook

**Status: source prepared; no MoLab job has run through this notebook.** Account access, actual notebook visibility and accelerator availability must be observed in the authenticated MoLab UI. No successful GPU allocation or model result is inferred from the provider documentation.

[Current status](EXECUTION_STATUS.json): authentication was not completed; the notebook has not been uploaded. The original Kaggle version 1 jobs subsequently completed and their actual [engineering results and timing warning](../kaggle_results_001/README.md) are preserved separately. The [independent wrapper review](INDEPENDENT_REVIEW.md) passed bounded local process/export checks; actual marimo execution remains unverified.

This is an interactive conversion of the already reviewed environment and adapter jobs. It preserves the exact source programs and embedded model/tokenizer archive. It adds a visible two-stage notebook, a bounded foreground supervisor and an independent artifact-recovery action. There is no headless API, SSH connection, daemon, control server, optimizer update or model-training stage.

## Files

| File | Purpose |
|---|---|
| `notebook.py` | Self-contained marimo notebook, including the exact public source payloads and supervisor |
| `supervisor.py` | Standard-library implementation of attempt guards, subprocess limits and evidence export |
| `build_notebook.py` | Source-only generator and embedded archive verification |
| `SOURCE_CHECKS.json` | Actual source/hash/AST checks; explicitly records that marimo/native execution is untested |
| `PORTABILITY_CHECKS.md` | Conversion decisions, remaining runtime checks and evidence limits |
| `INDEPENDENT_REVIEW.md` | Exact-byte review and four bounded synthetic process/export checks |
| `EXECUTION_STATUS.json` | Explicit provider access and NOT_RUN status |

## Preserved identities

| Program or artifact | SHA-256 |
|---|---|
| Environment probe | `70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719` |
| Reviewed adapter kernel | `c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81` |
| Embedded 33-member archive | `65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5` |

The unchanged inner source still contains Kaggle run IDs, a private flag, a T4 request and requested server timeout. Those fields describe the original source bundle. **They do not describe this MoLab execution.** The outer `cog.molab-interactive-supervisor.v1` receipt supplies the provider context, default public/unlisted visibility warning, measured accelerator inventory, actual status and notebook-enforced deadline. It explicitly says that the code cannot authenticate the hosting origin or assert private visibility.

## Interactive workflow

1. Obtain an authenticated, server-backed MoLab notebook through the supported UI. Open this notebook using the UI's available notebook import/editor flow; do not substitute a WebAssembly preview. Enable a GPU using the notebook-specs control and observe the actual sharing/visibility state.
2. Inspect the source summary and press **Run environment check**. This launches only the exact environment probe. It inventories packages/CUDA and checks an elementary analytic gradient; it loads no candidate model or corpus. The supervisor caps the job at five minutes, reserving five seconds for owned process-group teardown.
3. Review the displayed environment result and observed GPU. The adapter button is not created unless the environment receipt passes. The supervisor independently rechecks that the passing receipt comes from this notebook kernel and identifies real CUDA devices. A failed or stale receipt prevents model launch.
4. Press **Run adapter engineering** separately. The exact reviewed kernel performs its hash-pinned dependency bootstrap, existing correctness checks, and original FP32 capacity probe. It retains the original 125M per-model ceiling, width 512 / 12 layers / real 50,257-token engineering tokenizer / 1,024-token backward shape. No optimizer is created. The notebook caps this job at fifteen minutes, including a teardown reserve; the original core's 720-second alarm remains intact inside its own main-thread process.
5. After success, failure or interruption, press **Refresh and export evidence**. Download the per-stage ZIPs before ending the session. The export cell depends only on setup and its own button, so interruption of an execution cell does not make recovery depend on a missing result variable.

Only one attempt per stage is available through this notebook. Existing Kaggle queue cancellation is a separate operator task; this notebook neither authenticates to nor changes Kaggle jobs.

## Scope of the result

The checks establish engineering behavior and measured capacity on the observed runtime. A GPU's marketed specification is not a measurement. These models are untrained; backend-specific parameter counts differ. The output is not a language-quality result, architecture ranking, fair training comparison, production latency benchmark or scientific admission.

LocalOnly remains a separately parameterized control intended for independent future training. The frozen GPT-2-compatible engineering tokenizer does not reproduce the absent trained 32K nanochat tokenizer. Existing qualification, optimizer/data/trainer integration and competent-control requirements remain pending.

## Dependencies, files and visibility

The notebook's only frontend requirement is marimo. Its actual version is recorded. It does not upgrade Torch or infer availability from a dependency list. The environment probe reports installed versions; the unchanged child bootstrap installs only its two version/hash-constrained binary packages into a fresh attempt-local target directory. Runtime environment forwarding is restricted to driver/locale/certificate settings; credential variables, arbitrary Python search paths and pip index overrides are not forwarded. The wrapper disables pip configuration through `PIP_CONFIG_FILE` without modifying the immutable child source.

All embedded inputs are public source, notices, frozen public tokenizer bytes and inert fixtures. No user credential, private dataset, trained checkpoint or unrelated workspace file is included. No credential is requested by the notebook. MoLab's public/unlisted default must not be confused with a private Kaggle submission.

Each child gets a fresh working directory. The supervisor uses exclusive marker/result writes and never reuses an old attempt directory. The unchanged inner job may update its own progress receipts within that unique attempt; the wrapper preserves final or incomplete raw evidence. Export enumerates only specific receipts and a bounded log, never an entire directory. Truncated or skipped export content is declared in the artifact manifest.

## Persistence and cancellation limits

An exclusive local marker is written and flushed before launch. A JSON `mo.persistent_cache` claim records the current notebook-kernel owner. Reactive repeats encounter the marker; a cached claim from an older kernel is refused. These are safeguards for this reviewed attempt, **not a durable global lock**: MoLab may evict cached or attached storage. Never clear markers/caches to reroll a failure; a fresh attempt needs a separately recorded decision and reviewed identity.

The foreground supervisor owns a new child process group. Timeout, ordinary cell interruption and finalization terminate that group, including descendants after the group leader has exited, and reap the leader. Cleanup failure prevents PASS. Execution proceeds only while the visible run-button cell supervises its child. No process is intentionally left detached when the cell returns.

No user-space wrapper can promise final evidence or immediate cleanup after a hard notebook-kernel/VM kill. The original adapter core retains an internal deadline, but there is no new persistent watchdog service. Missing final receipts remain `INCOMPLETE_NO_FINAL_RECEIPT`; they do not prove compute stopped. The independent export action can capture whatever exists when the UI is available again. Downloaded evidence, not a scratch path or cache lifetime, is the handoff.

## Rebuild and review

From this directory, with the two existing public staged programs:

```sh
python build_notebook.py \
  --environment /workspace/scratch/7148f407b210/kaggle_submissions/environment001/probe.py \
  --kernel /workspace/scratch/7148f407b210/kaggle_submissions/adapter001/kernel.py
```

The generator verifies exact input hashes and all archive members, parses source syntax and writes only generated source/check artifacts here. It does not import marimo, Torch, tokenizer libraries or models, and it does not launch jobs. Changed helper/notebook bytes require a fresh exact-version wrapper review before execution.

Official references: [MoLab compute, sharing and storage](https://docs.marimo.io/guides/molab/), [run buttons](https://docs.marimo.io/api/inputs/run_button/), [reactive execution and interruption](https://docs.marimo.io/guides/reactivity/), [download controls](https://docs.marimo.io/api/media/download/), [persistent caching](https://docs.marimo.io/api/caching/), and [package management](https://docs.marimo.io/guides/package_management/). Runtime support and UI behavior still require observation; links are not execution receipts.
