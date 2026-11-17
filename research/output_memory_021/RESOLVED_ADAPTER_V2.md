# Supplied resolved-state adapter v2 (not frozen calibration)

`resolved_state.ResolvedExactState` is an explicitly engineered scientific-control proposal. It has the same392-dimensional interface as the raw calibration adapter, but exposes the effective resolved payload for **every** record, not only the queried identity. Original absent/value/reference tags and reference destinations remain. Current query identity is still parsed from external question text. No answer target or hidden operation annotation is read.

Sixteen first-appearance identity slots each contain3 type flags,3 normalized payload bytes, and16 reference flags;16 query flags follow;24 zeros pad to392. Dangling references, cycles and absent records have zero effective payload. Descriptor tags retain the distinction between absent and unresolved reference. COPY preserves the pre-operation effective snapshot. LINK resolves through current records, including deletion and recreation.

`receipt()` reports feature calls, all-record resolution count, traversed hops and synchronized feature-construction seconds. Evaluation must count this execution in complete-session cost. Features do not guarantee the neural output head can learn correct rendering, and this supplied grammar/state is not learned CSM competence.

The original `ExactState` and current calibration files are unchanged. This adapter is not automatically selected by the raw calibration or current cache/evaluation functions. A subsequent scientific source freeze must explicitly inject this class consistently into both cache construction and evaluation, record `adapter_version`, and include its execution costs. It replaces the supplied-state representation; it does not add a fit arm. Do not reuse raw-adapter caches for this version.

Four software tests pass: all-record resolution rather than query-only exposure; absent/dangling/delete/recreation; cycles/self-links; snapshot preservation and unchanged raw adapter. No model fit or scientific prediction was executed.
