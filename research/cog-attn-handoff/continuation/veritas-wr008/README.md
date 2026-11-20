# WR-008 CompanyWorld cache invalidation diagnostic

Research Lab OS froze two arms across seeds 71, 83 and 97. TrainLab executed six successful runs. A separate audit checked their receipts and rescored 1,152 saved submissions with Veritas CompanyWorld. The ecosystem quickstart also froze its own Forge dataset and passed reverification. The custom WR-008 inputs are generated from the pinned public fixture and frozen recipe, separate from that Forge validation split.

| Change | Unconditional reuse, full reward | Hash cited records, full reward |
| --- | ---: | ---: |
| Unchanged | 144/144 | 144/144 |
| Projection changed | 144/144 | 144/144 |
| Cited record revised | 0/144 | 144/144 |
| New authoritative record superseded existing citation | 0/144 | 0/144 |

Fresh evaluation succeeded on all cases. These constructed cases span shipments and approval limits. A replacement is designated by placing its record first in the public fixture and updating the verifier target. The control isolates invalidation of an already cached answer. It measures neither learned policy behavior nor latency. A guard for operational use would also need a trustworthy signal when the set of applicable authorities changes.

The original frozen attempts failed on a dictionary-access error in the harness. The corrected protocol has the v2 suffix. All failed receipts are retained. The audit ran under the same operator. Independent review and scientific qualification remain open.

## Reproduction

Install the four packages from FPC-effortless/veritas commit eec439492407b0aa26922c23b5ae40d95289f4eb following docs/ecosystem/README.md. From a fresh directory containing launch.py, audit.py and recipe/cache_trial.py, with the environment activated, run:

    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python audit.py
    veritas-ecosystem verify "$PWD/ecosystem"

The shipped database retains the frozen snapshots, worker history and recorded evidence. The custom driver pins the upstream Git revision and Research Lab snapshots the local recipe bytes. Runtime receipts contain original workspace paths.
