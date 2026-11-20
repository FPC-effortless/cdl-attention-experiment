import { readFileSync, writeFileSync } from 'node:fs';
import { reconcile, visibleAnchor, recheckAnchor } from '../issue-history-followup/anchor.mjs';
const rows=JSON.parse(readFileSync('/workspace/issue-history-followup/recipe/snapshots.json'));
const out=[];
const iso=t=>new Date(t*1000).toISOString();
for(const row of rows){
  const last=row.actions.at(-1);
  const timeline={issue:row.issue,state:last?.[1]?'closed':'open',lastActionAt:last?iso(last[0]):null,observedAt:row.replay_at};
  const snap=(s,t)=>({issue:row.issue,source:'github.issue_snapshot',createdAt:iso(row.created),updatedAt:s.updated_at,closedAt:s.closed_at,state:s.state,observedAt:t});
  const observed=snap(row.search,row.search_at),direct=snap(row.direct,row.direct_at);
  const r=reconcile(timeline,observed);
  const answer=r.disposition==='anchor'?visibleAnchor(r.anchor,{knowledgeAt:row.direct_at,validAt:row.direct_at}):r.state??null;
  out.push({issue:row.issue,split:row.split,raw:timeline.state,search:observed.state,truth:direct.state,answer,
    disposition:r.disposition,reason:r.reason,anchor:r.anchor,stable:JSON.stringify(row.search)===JSON.stringify(row.direct),
    versionConfirmed:r.anchor?recheckAnchor(r.anchor,direct):null,usableSequence:row.valid_sequence});
}
writeFileSync('results/snapshot.json',JSON.stringify(out));
