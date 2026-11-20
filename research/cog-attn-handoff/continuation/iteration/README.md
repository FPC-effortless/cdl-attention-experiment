# Structural no-ops, lazy temporal indexing and source-family verification

## Consequence training

With factorized source, destination and operation probabilities, let S_v and D_v be the summed source and destination probabilities for entities with state value v. Exact no-op likelihood is p(copy) times sum_v S_v D_v plus p(add) times S_0. Copy preserves a destination when both values agree. Modular addition preserves it exactly when the source value is zero. Grouped factor marginals give the exact negative-log-likelihood gradients in linear work for the four-value alphabet. Changed rows retain the 16-draw inverse-inclusion residual. The compatible mass estimator is unbiased there and the sampled log-gradient can still be biased.

Across reused paired seeds 13, 37, 61, held-pair accuracy rose from 92.42% to 94.14% (2892/3072). Checks fell from 24 to 17.93 per draw. Mean training times were 1.47 and 1.28 seconds respectively. On 747 held-pair no-ops at matched weighted checkpoints, 64 repeated residual samples per example gave mean gradient covariance trace 0.0257–0.0287 across seeds. The structural gradient has zero proposal variance and agreed with enumeration within 1.84e-15 per coordinate. This excludes minibatch variance.

## Temporal workload

At 32768 writes, five matched seeds, including construction, the persistent tree first beat the list at the 768-query prefix (167.8 versus 171.3 ms). At 512 queries the list won (124.6 versus 166.6 ms). The compact implementation packs per-entity valid-time buckets of width 256 in transaction order. On first full-bucket lookup for buckets of at least 64 records, it materializes a prefix-best array. Boundary buckets use a scan. At 4096 queries total mean time was 55.4 ms for buckets, 184.1 for the tree and 791.9 for the list. All 20480 bucket and tree answers matched the list and 640 sampled list answers matched a full scan. At seed 301 packed bytes were 488888, 6553612 and 393216 respectively (the list figure is a packed estimate). These exclude container allocations. The bucket method can scan many buckets on sparse histories. Timings use fixed implementation order on a shared machine.

## Verification

Five seeds, 40960 high-criticality trials per regime, up to seven reads from four known families, two successive reads per family until budget exhaustion. A family repeats its initial signal. For a 0.998 posterior threshold and known 15% marginal error, family-aware verification counts one vote per distinct family. Under independent family errors, pooled verification admitted 30591 with 926 wrong and the family rule admitted 21425 with 25 wrong. Under shared corruption across all families both admitted 40960 with 6088 wrong. Family labels and within-family correlation were supplied. This pilot tests gating, not acquisition of an independent channel.

## Replay

Python 3.12 and NumPy 2.3.5. Verify the shipped files with sha256sum -c SHA256SUMS before replay. From the extracted directory in a copy run the following.

    python run.py
    python gradient_audit.py
    python systems.py

The package contains the six checkpoints, vocabulary and alias maps, raw per-seed metrics, protocol, source and checksums. The benchmark continues the WR-005–007 subsystems. End-to-end WR-008–010 runtime qualification remains open.
