# INFO-002 secondary check: follow the edit only for the right entity

With the exact same one-token edit and history, ask one question about the edited
entity and one about the unchanged entity. A pair passes only if the first
answer follows the edit and the second preserves the old value. The four-answer
measure requires both separate one-token edits to pass on the same base history.
All answers must be unique maxima of the full 287-token output.

| Frozen checkpoint | Physical edit | Natural edited input | Value-tap-only transplant |
|---|---|---:|---:|
| Weak 1371 | First clause | 91/128 | 91/128 |
| Weak 1371 | Second clause | 91/128 | 93/128 |
| Weak 1371 | Both separate edits, four answers | 84/128 | 86/128 |
| Strong 1171 | First clause | 126/128 | 126/128 |
| Strong 1171 | Second clause | 123/128 | 125/128 |
| Strong 1171 | Both separate edits, four answers | 122/128 | 124/128 |

The weak checkpoint's 33 incorrect adoptions of the inserted value on the
other-entity query all followed edits to the **first** clause, and every one
still answered the edited-entity query correctly. Both natural and tap-only
passed the same 91 first-clause pairs. For second-clause edits, the tap-only
condition eliminated three wrong-query insertions but also got one fewer
edited-entity question right. The net two additional pairs hide four gains and
two losses. The paired measure exposes the selectivity problem that target-only
accuracy cannot see.

This is a **post-hoc score-only reanalysis** of the same sealed INFO-002 logits.
No model calls, fits, or new data. The locked primary contrast and binding
qualification are unchanged. These counts do not establish a learned repair.
The score source and rule are linked in this directory; the exact logits and
checkpoint bytes remain private. The private structured result SHA-256 is
`aae8df118ff97f800b857d3f48e72d26811202a2752d2420b4026dabb54bdc2f`.
