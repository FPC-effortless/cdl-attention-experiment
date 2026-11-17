# CSM-PARALLEL-022: reviewed software-calibration source backup

Separate from completed OUTPUT-MEMORY-021. No original CSM-PARALLEL-001 archive was located; this is not a reproduction of its claims or an equivalent native CSM implementation.

The new candidate combines learned byte parsers (Transformer/GRU) with an explicit typed executor supporting OpenMP parallel independent banks and stable within-bank operation order. Cross-bank dependencies fall back to serial execution. These memory semantics and scheduling rules are supplied engineering, not learned capability. Read EXECUTOR_MATH.md and the provenance/scope documents; mathematical reasoning and parity tests are not a newly kernel-verified Lean proof.

Independent review approves only prospective bounded CPU software calibration: 16 optimizer updates on constructed fixtures, 64 parser scheduling measurements and 24 engineered executor measurements. No scientific fits, trained quality result, protected TEST or speed superiority is claimed. Random-model fixed-order timing is a resource estimate; integer query completion is not complete natural-language output. Compilation/loading costs must be distinguished from warmed calls.

PUBLIC_SOURCE_SHA256.json records exact published bytes. Private input/runtime/admission/review manifests, dependency vendor, compiled artifacts and evidence logs are excluded. This is source recovery, not a self-contained admitted execution closure. Execution still requires reviewed native configuration, owner-authorized private backup and exact runtime bindings. Preserve failures, use Forge/TrainLab/Veritas/Research Lab, no paid resources. Draft publication does not authorize merge or bypass checks.
