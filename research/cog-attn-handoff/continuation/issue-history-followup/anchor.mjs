// Reconcile a snapshot and a recorded stream for one issue.

// A snapshot anchor carries a valid time and a distinct knowledge time.

function instant(value) {
  return typeof value === "string" ? Date.parse(value) : Number.NaN;
}

function reconcile(timeline, snapshot) {
  const sameRecord = timeline.issue === snapshot.issue;
  const states = ["open", "closed"];
  const observed = Math.max(instant(timeline.observedAt), instant(snapshot.observedAt));
  const created = instant(snapshot.createdAt);
  const updated = instant(snapshot.updatedAt);
  const closed = instant(snapshot.closedAt);
  const lastEvent = timeline.lastActionAt == null ? -Infinity : instant(timeline.lastActionAt);
  const validInput =
    sameRecord && snapshot.source === "github.issue_snapshot" &&
    states.includes(timeline.state) && states.includes(snapshot.state) &&
    created <= updated && updated <= instant(snapshot.observedAt) &&
    created <= instant(timeline.observedAt) &&
    (lastEvent === -Infinity ||
      (created <= lastEvent && lastEvent <= instant(timeline.observedAt))) &&
    Number.isFinite(observed) &&
    (snapshot.state === "closed" ? created <= closed && closed <= updated : snapshot.closedAt === null);
  if (!validInput) return { disposition: "defer", reason: "invalid_provenance" };
  if (snapshot.state === timeline.state &&
      (snapshot.state === "open" || lastEvent === closed)) {
    return { disposition: "consistent", state: timeline.state };
  }
  if (snapshot.state === "closed" && closed > lastEvent) {
    return {
      disposition: "anchor",
      anchor: {
        issue: timeline.issue,
        state: "closed",
        validFrom: new Date(closed).toISOString(),
        knownFrom: new Date(observed).toISOString(),
        source: snapshot.source,
        snapshotUpdatedAt: snapshot.updatedAt,
      },
    };
  }
  return { disposition: "defer", reason: "unresolved_state_history" };
}

function visibleAnchor(anchor, { validAt, knowledgeAt }) {
  const validFrom = instant(anchor.validFrom);
  const knownFrom = instant(anchor.knownFrom);
  if (!Number.isFinite(validFrom) || !Number.isFinite(knownFrom)) return null;
  return instant(validAt) >= validFrom && instant(knowledgeAt) >= knownFrom
    ? anchor.state : null;
}

function recheckAnchor(anchor, nextSnapshot) {
  return anchor.issue === nextSnapshot.issue &&
    anchor.source === nextSnapshot.source &&
    nextSnapshot.state === anchor.state &&
    instant(nextSnapshot.closedAt) === instant(anchor.validFrom) &&
    instant(nextSnapshot.updatedAt) === instant(anchor.snapshotUpdatedAt) &&
    instant(nextSnapshot.observedAt) >= instant(anchor.knownFrom);
}

function offset(iso, milliseconds) {
  return new Date(instant(iso) + milliseconds).toISOString();
}

function verifyRows(cases) {
  const checked = [];
  for (const { timeline, archived, latest } of cases) {
    const result = reconcile(timeline, archived);
    if (result.disposition !== "anchor") throw Error(`Expected anchor for ${timeline.issue}`);
    const a = result.anchor;
    const checks = [
      { validAt: offset(a.validFrom, -1), knowledgeAt: a.knownFrom, expected: null },
      { validAt: a.validFrom, knowledgeAt: offset(a.knownFrom, -1), expected: null },
      { validAt: a.validFrom, knowledgeAt: a.knownFrom, expected: "closed" },
      { validAt: latest.observedAt, knowledgeAt: latest.observedAt, expected: "closed" },
    ];
    for (const q of checks) {
      if (visibleAnchor(a, q) !== q.expected) throw Error(`Anchor cutoff mismatch for ${timeline.issue}`);
    }
    if (!recheckAnchor(a, latest)) throw Error(`Fresh snapshot disagrees for ${timeline.issue}`);
    if (a.eventId || a.actor) throw Error(`Snapshot claimed an event identity for ${timeline.issue}`);
    checked.push({ issue: timeline.issue, anchor: a, cutoffChecks: checks.length,
                   liveMatch: true, liveStateReason: latest.stateReason });
  }
  return {
    knownDiscrepancyCases: cases.length,
    correctStreamOnlyStates: cases.filter(r => r.timeline.state === r.latest.state).length,
    confirmedSnapshotAnchors: checked.length,
    passedCutoffChecks: checked.reduce((n, r) => n + r.cutoffChecks, 0),
    checks: checked,
  };
}

function checkInvariants(cases, report) {
  const first = cases[0];
  const timeline = first.timeline;
  const snapshot = first.archived;
  const anchor = report.checks[0].anchor;
  const checks = [
    reconcile({ ...timeline, state: "closed", lastActionAt: snapshot.closedAt }, snapshot).disposition === "consistent",
    reconcile({ ...timeline, lastActionAt: offset(snapshot.closedAt, 1000) }, snapshot).disposition === "defer",
    reconcile({ ...timeline, state: "closed", lastActionAt: snapshot.closedAt },
              { ...snapshot, state: "open", closedAt: null }).disposition === "defer",
    reconcile(timeline, { ...snapshot, source: "unknown" }).reason === "invalid_provenance",
    reconcile(timeline, { ...snapshot, observedAt: offset(snapshot.updatedAt, -1000) }).reason === "invalid_provenance",
    !recheckAnchor(anchor, { ...first.latest, updatedAt: offset(first.latest.updatedAt, 1000) }),
    reconcile({ ...timeline, observedAt: offset(snapshot.createdAt, -1000) }, snapshot).reason === "invalid_provenance",
    visibleAnchor({ ...anchor, validFrom: "invalid" },
                  { validAt: first.latest.observedAt, knowledgeAt: first.latest.observedAt }) === null,
  ];
  if (!checks.every(Boolean)) throw new Error("Snapshot reconciliation invariant failed");
  return checks.length;
}

export { reconcile, visibleAnchor, recheckAnchor, verifyRows, checkInvariants };
