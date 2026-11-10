# Resumption record, November 10

Original processes no longer existed when the user requested resume. All three
immediate fits completed and remain untouched. Single-timescale checkpoints were
saved at update128 for all three seeds. Restore exact model, optimizer, RNG and
registered schedule, and continue at update128, an episode boundary. No data,
optimizer settings, endpoint or model selection changes. Lost unsaved computation
is not presented as measured training compute. Historical per-update losses and
wall time were not in those checkpoints; unavailable entries are null and total
training time is unknown for the resumed fits. New checkpoints save this telemetry.
Protocol remains byte-identical. This is continuation, not a new seed trial.
