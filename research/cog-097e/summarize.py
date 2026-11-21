from pathlib import Path
import json,statistics
rows=[json.loads(Path(f'result-{i}.json').read_text()) for i in (1701,1707,1711)]
summary=dict(name='COG-ATTN-097e outcome-trained read-only retriever, not full COG',
  seeds=[1701,1707,1711],
  reused_distinct_test_requests=184,
  total_seeded_evaluations=552,
  training_episodes=sum(x['event_receipt']['episodes'] for x in rows),
  training_actions=sum(x['event_receipt']['actions'] for x in rows),
  training_successes=sum(x['event_receipt']['successes'] for x in rows),
  methods={})
for name in rows[0]['results']:
    variants=[r['results'][name] for r in rows]
    timings=[r['times'][name]['test'] for r in rows]
    summary['methods'][name]={
        k:sum(v[k] for v in variants)
        for k in ('correct','wrong','abstained','accepted','unknown_false','always_correct')}
    summary['methods'][name]['precision']=summary['methods'][name]['correct']/max(1,summary['methods'][name]['accepted'])
    summary['methods'][name]['coverage']=summary['methods'][name]['accepted']/552
    summary['methods'][name]['mean_online_ms_per_query']=round(statistics.mean(v['online_ns']/184e6 for v in timings),3)
    summary['methods'][name]['mean_index_build_ms']=round(statistics.mean(v['build_ns']/1e6 for v in timings),3)
Path('SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2))
