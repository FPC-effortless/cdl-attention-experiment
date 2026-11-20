# Exploratory reuse of TrainLab competence panels on CSM-PARALLEL-022

18 November 2026. No model execution, training, new inference or changes to original gates.

## Provenance and implementation
Recovered the original private evidence package from Library and verified its original SHA256 `81c0b529ebab0ebfc230e5d3b615861a8a38f1ccddf4242e1f25b18c8c0fd685` and all manifest entries. Used the actual merged PR539 implementation from Veritas commit `db9075ec3fa4c4df1a5a02ea993b0fb9ef548b50`. Its 16 focused tests pass. The byte-exact original adapter `adapt.py` is published here. Its 44 compressed panel/observation/report bundles and aggregate `SUMMARY.json` remain private.

Each panel binds actual seed checkpoint hashes, frozen dataset hashes, original prediction-file hashes, adapter/source identity, query membership and expected answers. The native scorer directly consumes supplied observations extracted from saved predictions, not independently executed models. Changing an expected answer without updating its external panel hash was rejected in all 44 panel mutation checks. Plans were frozen now, after outcomes were already known: this is not prospective preregistration.

## Exact match to existing metrics
For each of two architectures and seven axes, three different explicit panels cover all three seeds:

1. Query panels bind expected finite JSON `{query:true, valid:true, value:expected_payload_or_null}`. Observations bind the actual predicted QUERY flag, action validity and payload. This preserves the original distinction between a legitimate absent answer and a failed/malformed query.
2. Whole-session panels contain all six or twenty-four query members plus an explicit `no_unsolicited` guard, reproducing the original rule that a spurious QUERY at a non-query event invalidates a clean session. Guard rows are not model queries. An independent check found no actual complete-map count difference between guarded and unguarded scoring in these saved results; the guard preserves the original definition rather than explaining an observed discrepancy. Their augmented query denominator (e.g. 9,600 instead of 9,216 on the independent-region panel) is not reported as query accuracy; only their map metric represents session accuracy.
3. Paired-first panels put the two variants' first queries in one map, reproducing the original paired-first criterion separately.

All 42 architecture/axis/seed-cell sets reproduce the corresponding original query counts, complete sessions and paired-first counts exactly. The competence API's own illustrative 95% query/map statuses do not replace 022's registered 95% complete-session and paired-first gates. Both original gates remain failed overall.

The API handles variable map cardinality correctly: six queries in ordinary sessions and twenty-four in independent-region sessions. It is bounded to 10,000 cases, so the adapter uses one architecture/axis per panel rather than pooling the entire campaign. The largest panel has 9,600 members including guards.

## Exploratory regional breakdown
Splitting each independent-region session into its four six-query region maps yields:

| Parser | Correct query answers | Correct original 24-query sessions | Correct 6-query region maps |
|---|---:|---:|---:|
| Transformer |890/9,216|0/384|0/1,536|
| GRU |409/9,216|0/384|0/1,536|

Thus the observed zero whole-session result does not disappear merely by scoring six-query regional subsets instead of the conjunction of 24 queries. This is descriptive, not a causal attribution. Regional maps are correlated within session; the same 128 sessions/64 pairs are reused across three seeds. The 1,536 regional evaluations are not 1,536 independent experimental subjects.

The documented compound-shift limitation remains: independent regions also introduce unseen bank0–bank3 prefixes, sixteen records and interleaving; cross-bank cases also change basenames and east/west prefixes. Neither this rescoring nor the tool isolates those factors. No broader competence, new causal mechanism, scientific promotion or model superiority follows.

## Reproduce with authorized private artifacts

The published adapter preserves its original bytes and adapter hash. It intentionally expects a relative private restoration layout beside `adapt.py`: `veritas539/integrations/trainlab/src` from Veritas commit `db9075ec3fa4c4df1a5a02ea993b0fb9ef548b50`; `admitted/seed11`, `admitted/seed23`, `admitted/seed37` containing the original data and INPUTS manifests; and `seed11/runs`, `seed23/runs`, `seed37/runs` containing the single intended output stage per seed. Verify exact source and artifact identities against the private manifests first. This is not a standalone public dataset or model runner.

Run `python adapt.py` only in a fresh owner-authorized private restoration. It writes `reports/` and `SUMMARY.json` alongside itself. Generated bundles contain private case membership and observations; do not publish them. The public source grants no access to private artifacts and authorizes no model execution, new training or change to the original gates.
