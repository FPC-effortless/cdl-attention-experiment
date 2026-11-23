# FACT-006: Use the verified record, not the model’s guess

The previous guard rejected a request when the learned router chose the wrong record. This test changed one thing: if the current registry could resolve exactly one authorized record, the transaction used that record directly. Both arms still used the same frozen learned Keep/Drop and A/B value heads, and both rejected unknown names, inactive or ambiguous aliases, and stale versions.

| Held outcome | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Entire sessions exact, verify → authoritative | 22/24 → 24/24 | 24/24 → 24/24 | 24/24 → 24/24 |
| Legitimate writes exact, verify → authoritative | 92/96 → 96/96 | 96/96 → 96/96 | 96/96 → 96/96 |
| Legitimate reads exact, verify → authoritative | 256/264 → 264/264 | 264/264 → 264/264 | 264/264 → 264/264 |

Starting from **identical, correct board state**, the change rescued both of the two naturally misrouted value-changing writes and both misrouted reads available for this comparison. All four direct rescues occurred in one seed, across two sessions, with two and eight records respectively. The other two seeds had no qualifying disagreement, so they do not establish recovery under drift. The affected sessions also gained exactness later in their trajectories, but those later operations are not matched-state rescues. No previously correct matched-state outcome was lost, and neither mode made an unauthorized commit or issued a wrong-identity answer on this panel.

Every authoritative session stayed exact after a real cold process restart. Workers received current operations and read restored typed board state, not old raw turns. This supports exact registry routing as a way to reduce safe-but-unhelpful refusals in this prototype. It does not demonstrate learned alias resolution, repair to the learned router, general authorization, faster inference, or a connection to the original million-parameter model’s language output.
