# First-party provenance checklist

All checks below are necessary. An authorized researcher must record the decision of an independent reviewer before moving between metadata, DESIGN and any later DEV stage. An unmet check preserves the hold.

| Check | Accept | Refuse or keep the hold |
| --- | --- | --- |
| Origin | USGS Water Data OGC API v1 on `api.waterdata.usgs.gov`, or an approved USGS-published export with proved identical series and selection | Mirror, client code, screenshot, copied values, transformed bundle without an origin receipt, or an unapproved alternate transport |
| Exact series | Three frozen station IDs, one primary instantaneous `00065` series per site, exact time-series ID and reviewed coverage, datum epoch, units, cadence and producer | Guessing an ID/datums, ambiguous series, substituting a station after the freeze, or treating software examples as source data |
| Rights | USGS-origin publication and source-series usage review, with exact citation and any third-party restrictions retained | Assuming every file on an agency site inherits the agency's public-domain policy |
| Preservation | Immutable bytes or documented first-party export, exact canonical request/selector, UTC capture time, response status and byte/row caps, SHA-256 and separate metadata receipt | Snippets or rendered summaries, unknown truncation, silent retries, checksum mismatch, duplicate time/series keys |
| DESIGN boundary | `[2023-10-02T00:00:00Z, 2024-01-22T00:00:00Z)`, `00065`, no latest/unbounded query, ≤12,000 rows per station, ≤36,000 total and ≤64 MiB | Any DEV/TEST, post-window rows, unaudited redirect, model fit, prediction, public alert, or support counts inferred from metadata |
| Scientific use | `Approved` observed readings, known units/flags; aggregate support screen only; separately approve a future DEV recipe | Treating missing/estimated data as correct outcomes or treating archive record modification time as live availability |
| Verification scope | Independent operation and exact custody for a prospective shadow; separately operated real workflow and genuinely separate team for interactive/replication claims | Same-source copies, same-operator second machines, internal re-scoring, simulations or public API results claimed as independent physical outcomes |

If a permitted raw ingress can be arranged, check official metadata first. If no exact eligible series is verified, remain at `HOLD_SOURCE_INGRESS`. If verified DESIGN bytes fail the frozen support screen, record `HOLD_DESIGN_SUPPORT` with the exact failed item; do not replace a site/window or lower the floor to force a run. Full 070 remains distinct even if offline DESIGN and DEV eventually pass.
