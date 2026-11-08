# Original LRC repair publication

This one-time publication payload preserves the recovered original Git objects,
including their commit identities. It publishes only:

- `research/lrc-route-repair-002-20261108` at
  `3f71af6eb53827eb7eb3c74e10964f70e8e84460`.
- `research/lrc-route-repair-002-handoff-20261108` at
  `566ba9933a35cf88a55f830891d8d6e33d6ca524`.

The handoff branch contains the original checkpoint/evidence ZIP, source patch,
comparison/dependency ZIP, SHA-256 manifest, and restore instructions under
`artifacts/lrc-route-repair-002/`. The frozen source commit
`6c3976efc2382e28731009f9637ecf31872e5c13` is preserved in the repair history.

`publish.py` verifies the split Git bundle, its prerequisites, both original
heads, repository connectivity, and the original artifact hashes. Its default
mode uses a temporary repository and does not publish. `--publish` runs only in
the narrowly scoped GitHub Actions job, refuses a differing existing branch,
and atomically creates missing refs with leases requiring them to be absent.
It never changes the default branch, merges code, executes model code, or
repeats training. No external secrets are required; the job uses only its
repository-scoped `GITHUB_TOKEN`.

After both branches are verified remotely, the job creates a prerelease at the
original handoff commit and uploads the original artifacts and restore helpers.
It never clobbers an existing release or asset. A release-name conflict stops
the release step for inspection; already-published branches remain valid.

Publication is an artifact recovery operation, not new research validation.
GitHub token pushes do not automatically trigger ordinary downstream CI.
