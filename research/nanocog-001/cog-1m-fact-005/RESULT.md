# FACT-005: A durable identity guard stopped unauthorized learned writes

A deterministic identity check ran inside the same SQLite transaction as the learned route, scoped version check, and fact update. On 24 fresh held sessions per seed with 2, 3, 4 or 8 records, the guard prevented all observed unauthorized commits and wrong-identity reads. Qualification was **2 of 3 seeds**, not a full pass: one seed missed the 95% legitimate-write requirement at 91/96 (94.79%).

| Held outcome | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Unauthorized durable commits, control → guard | 26 → 0 | 27 → 0 | 24 → 0 |
| Wrong-identity reads, control → guard | 80 → 0 | 83 → 0 | 72 → 0 |
| Guard blocks control commits from identical pre-state | 24 | 24 | 24 |
| Exact legitimate guarded writes | 92/96 | 91/96 | 96/96 |
| Exact legitimate guarded reads | 256/264 | 252/264 | 264/264 |

A committed version counts even if the value did not change. Of the 72 paired unauthorized control commits stopped by the guard, 60 would have changed a value. Unknown names, inactive aliases and ambiguous aliases were rotated so each got an equal first challenge. On matched clean histories the guard did not reduce exact legitimate read or write counts relative to the control. After divergence, later commands sometimes had stale versions; the full trajectory count is not the paired intervention count. At eight records the weakest seed safely refused three legitimate wrong routes; those refusals made two later valid writes stale, and one later read returned an outdated value on the correct identity.

Both arms used the same three frozen FACT-003 checkpoints and the same learned selection, Keep/Drop and value projections. Unique exact aliases were resolved by an explicit directory in both arms. The guard refused rather than substituting an answer when the learned address differed from the authorized canonical identity. Same-value authorized Keep still advanced the scoped version; cancellation and stale commands did not.

Fresh isolated workers reopened durable typed SQLite state and received only the current command, not old raw turns. They still read and re-encoded board facts. This establishes a bounded integration result in a dedicated prototype, not learned alias understanding, strong caller authentication, production archive integration, crash/power-loss tolerance, or a serving-speed advantage.
