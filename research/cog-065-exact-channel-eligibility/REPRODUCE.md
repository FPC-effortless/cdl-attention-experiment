# Reproduce the mechanics, not an approval
Use Python 3 and the Forge, TrainLab and Research Lab integrations from Veritas commit eec439492407b0aa26922c23b5ae40d95289f4eb. The local source `primitive_reference.py` is an unchanged copy of the earlier COG-066 simulated primitive domain; no learned options or planner changes are imported.

`python test_clock.py` runs hand-authored chronology and receiver-safety mechanics. The generator and two guarded native stages are included to inspect the method. A new experiment needs an independently approved manifest, pinned runtime and immediate stage-specific custody checks; the original private reviews and data are deliberately not shipped. Do not reuse the public report as a launch approval. No learner should run from this packet.
