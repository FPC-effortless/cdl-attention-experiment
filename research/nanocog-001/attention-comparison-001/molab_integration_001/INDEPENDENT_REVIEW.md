# Independent MoLab wrapper review

**Disposition: suitable to preserve as a prepared notebook; actual MoLab execution remains unverified.** No blocking source defect was found within this review's scope. This is neither a successful GPU run nor admission of a scientific attention comparison.

## Exact inspected sources

| File or payload | SHA-256 |
| --- | --- |
| `supervisor.py` | `59b993b42176370380e1bd2d68fb0ef7f588f0f6efc8ffa07028a4210d229cfb` |
| `build_notebook.py` | `1e666165ec1abd9ef588e56f9ab4f68f639cd33c0730c896abdddb675642b5fe` |
| `notebook.py` | `ec79f2d856070caa1de574a2acc0d7e8d41a96e1fa50d135bd47db75ef88bc31` |
| Embedded environment worker | `70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719` |
| Embedded adapter worker | `c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81` |
| Embedded adapter archive | `65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5` |

The three Python files passed standard-library AST parsing. Independent extraction from the notebook confirmed the exact helper bytes and both worker hashes. All 33 archive entries matched the embedded manifest's names, lengths, and SHA-256 values; entries were bounded regular files with relative paths and no parent traversal. `SOURCE_CHECKS.json`, `PORTABILITY_CHECKS.md`, and the README were inspected alongside the source.

## Source findings

- Opening the notebook can prepare its helper and UI, but worker launch is behind explicit run buttons. The environment and adapter budgets are 300 and 900 seconds respectively, with five seconds reserved for cleanup.
- The adapter requires a successful environment receipt from the same notebook kernel, including CUDA availability and per-device analytic checks. The adapter button also depends on the environment pass.
- The cached claim checks kernel ownership, and an exclusive local marker is created before child launch. These prevent ordinary repeat attempts while the corresponding storage remains available; they are not a durable global execution lock.
- The worker runs in a new POSIX process group. Cleanup executes in `finally` and attempts TERM and KILL against that group even when its leader has already exited. Cleanup exceptions produce failure evidence rather than a success status.
- Final evidence is written locally before returning or re-raising an ordinary interruption. The separate export button has no dependency on either run-result variable and can recover allowlisted artifacts after interrupted reactive cells. Export enumerates bounded files rather than archiving the surrounding notebook directory.
- Historical Kaggle metadata is identified as historical. The outer receipt does not claim verified hosting origin, private visibility, a T4, training, or a scientific benchmark result.

## Bounded local checks actually performed

Only standard-library code ran. In an isolated temporary directory, the reviewed supervisor was imported inertly; worker decoding/launch input was replaced **in memory** with synthetic Python processes, marimo package-version lookup was stubbed, and the test budget was reduced to six seconds. The original worker programs were never executed or imported. No source file was modified for these checks.

| Synthetic case | Observed receipt status | Elapsed seconds |
| --- | --- | ---: |
| Leader exits while a descendant remains | `FAILED_MISSING_OR_INVALID_RECEIPT` | 0.408 |
| Worker exceeds the reduced test budget | `TIMEOUT` | 3.009 |
| Supervisor receives `KeyboardInterrupt` | `INTERRUPTED` | 2.258 |
| Cleanup reporting raises a synthetic exception after actual cleanup | `FAILED_PROCESS_CLEANUP` | 0.405 |

Each case left no descendant executing (the process was absent or a non-running zombie), created a final evidence ZIP, refused a repeat attempt, and supported independent export. Removing the synthetic final receipt resulted in `INCOMPLETE_NO_FINAL_RECEIPT`, not a success claim. Missing adapter prerequisites and a stale kernel-owner claim were separately rejected. The ordinary-exit case intentionally produced no environment observation, so its failure status is the expected result. The cleanup-exception case verifies failure reporting and export; it does not demonstrate that an OS-denied signal can terminate a process.

## Remaining execution gates and limitations

The local environment has no installed marimo package. Marimo parser/reactivity behavior, `persistent_cache` compatibility, MoLab authentication, notebook visibility, actual CUDA hardware, dependency installation, and both original engineering jobs remain untested by this review. No authentication, credentials, remote calls, model imports, native numerical execution, or optimizer updates were performed.

The monotonic supervisor budget is an application-level timeout with scheduling and teardown overhead, not a provider-enforced hard compute deadline. A hard kernel/VM termination can prevent cleanup and evidence finalization; disappearing MoLab cache/storage can invalidate persistence assumptions. The documented requirement to download evidence before ending the session remains necessary.

Proceeding from source preparation to an execution claim requires successful interactive marimo/MoLab validation and actual downloaded receipts. The local synthetic checks above do not replace those observations.
