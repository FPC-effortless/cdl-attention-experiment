import json,statistics,math
from pathlib import Path
from experiment import HERE,ARMS

def main():
 rows=[json.loads((HERE/f'seed-{seed}'/f'{arm}-result.json').read_text()) for seed in (11,23,37) for arm in ARMS]
 for seed in (11,23,37):
  rr=[r for r in rows if r['seed']==seed]
  assert len({r['initial_parameters_sha256'] for r in rr})==1
  assert len({r['schedule_sha256'] for r in rr})==1
 assert {r['parameters'] for r in rows}=={34479}
 assert {r['training_targets'] for r in rows}=={262144}
 summary={}
 for arm in ARMS:
  rr=[r for r in rows if r['arm']==arm];ns=[r['evaluation']['none']['nll'] for r in rr]
  summary[arm]=dict(mean_nll=statistics.mean(ns),min_nll=min(ns),max_nll=max(ns),seed_nll=ns,mean_accuracy=statistics.mean(r['evaluation']['none']['accuracy'] for r in rr),mean_ece=statistics.mean(r['evaluation']['none']['ece'] for r in rr),mean_train_nll=statistics.mean(r['train_first_episode']['nll'] for r in rr))
  for intervention in ('trace_reset','trace_swap','record_reset'):
   summary[arm][intervention+'_post256_delta']=statistics.mean(r['evaluation'][intervention]['post256_nll']-r['evaluation']['none']['post256_nll'] for r in rr)
 paired={}
 for arm in ARMS[1:]:
  paired[arm]=[next(r['evaluation']['none']['nll'] for r in rows if r['seed']==s and r['arm']==arm)-next(r['evaluation']['none']['nll'] for r in rows if r['seed']==s and r['arm']=='instant') for s in (11,23,37)]
 (HERE/'summary.json').write_text(json.dumps(dict(arms=summary,paired_nll_minus_instant=paired,matched_initialization=True,matched_schedules=True),indent=2)+'\n')
 lines=['# CSM-LONGBENCH-003 results','', 'Twelve final checkpoints; three seeds; four equal-capacity controls. Each fit used 262,144 training targets, 32 times the initial pilot. Final DEV covers 32,768 targets distributed across the exposed split. No protected-test scoring.','', '| Arm | Mean NLL ↓ | Seed range | Accuracy | ECE ↓ |','|---|---:|---|---:|---:|']
 for arm,v in summary.items():lines.append(f"| {arm} | {v['mean_nll']:.6f} | {v['min_nll']:.6f}–{v['max_nll']:.6f} | {v['mean_accuracy']:.2%} | {v['mean_ece']:.5f} |")
 lines+=['','## Paired differences from immediate','', 'Negative NLL differences favor the candidate. Three seeds do not support precise uncertainty estimates.','', '| Arm | Seed11 | Seed23 | Seed37 |','|---|---:|---:|---:|']
 for a,v in paired.items():lines.append('| '+a+' | '+' | '.join(f'{x:+.6f}' for x in v)+' |')
 lines+=['','## Memory interventions','', 'Post-token256 NLL change relative to the untouched rollout. Positive means the intervention hurts. Trace swap is a one-time cyclic swap across different real-text streams. Record reset zeros values, preserving links, cursor and trace. These out-of-distribution interventions measure total downstream effects, not isolated semantic memory.','', '| Arm | Trace reset | Trace swap | Record reset |','|---|---:|---:|---:|']
 for a,v in summary.items():lines.append('| '+a+' | '+' | '.join(f"{v[k+'_post256_delta']:+.6f}" for k in ('trace_reset','trace_swap','record_reset'))+' |')
 lines+=['','## Reproduction and boundaries','', '`experiment.py --seed 11` (also23/37) repeats fits. `test_contracts.py` verifies split and state contracts. `replay_and_benchmark.py --timing` independently reloads final checkpoints and measures isolated CPU steps; run after training exits. All checkpoint hashes and source/data/schedule identifiers are preserved.','', 'This is an exposed-DEV study on one small real corpus, with fixed budget and no tuning. The longer runs also broaden the sampled training data, so improvement over the old pilot cannot be attributed to update count alone. The three main arms have equal learned-array shapes and nominal operation counts; hardware FLOPs are unmeasured. The tb64 arm changes backward truncation and thus does not have an identical backward computation graph. CPU training ran in concurrent independent processes; throughput is not a clean architecture speed comparison. Deployment performance requires an optimized runtime.','', 'Historical pilot CI failed its exact cross-host floating-point assertion without reporting a delta; its root cause is unconfirmed. The revised replay reports numerical errors with absolute tolerance1e-6, registered before these fits. A subsequent independent run passed all nine pilot replays with zero observed error. Historical failure records remain available. Resumed single-timescale fits have missing early per-update loss/time telemetry; those entries are null, not reconstructed.']
 baseline=[json.loads((HERE/f'seed-{s}'/'baselines.json').read_text()) for s in (11,23,37)]
 lines+=['','## Interpretation and reference baselines','',
 'The strongest mean result is multi_tb64, but its advantage is seed-dependent. The longer 256-token gradient window does not improve the multiple-timescale condition here. Clearing the multiple-timescale trace slightly improves mean post256 loss, so this result does not demonstrate that beneficial retained history explains the gain.',
 '',f"Train-only unigram mean NLL: {statistics.mean(b['unigram_nll'] for b in baseline):.6f}; bigram: {statistics.mean(b['bigram_nll'] for b in baseline):.6f}. These count controls see the same sampled targets, but are not parameter- or compute-matched. The bigram has 65,536 count cells. They establish practical reference performance, not a fair architecture-superiority claim."]
 diag=json.loads((HERE/'prediction_diagnostics.json').read_text())['rows']
 lines+=['','## Exploratory post-fit collapse diagnostics','',
 'All 12 models choose ASCII space as their most frequent top-1 prediction. Its prediction share ranges from '+f"{min(r['most_predicted_share'] for r in diag):.2%} to {max(r['most_predicted_share'] for r in diag):.2%}. "+'NLL can improve while the most likely token barely changes; this is why the loss improvement is not yet useful language competence.',
 '', '| Seed | Arm | Space prediction share | Distinct top-1 bytes | Effective destinations | Effective operators |','|---:|---|---:|---:|---:|---:|']
 for r in diag:lines.append(f"| {r['seed']} | {r['arm']} | {r['most_predicted_share']:.2%} | {r['distinct_predicted_bytes']} | {r['effective_destinations']:.2f} | {r['effective_operators']:.2f} |")
 lines+=['','Effective counts are exp(entropy) of selection frequencies, not proof of semantic specialization. Several runs effectively use only one operator or destination. Complete raw frequencies and routing margins are in prediction_diagnostics.json. These checks were added after inspecting the first completed fits and are explicitly exploratory.']
 timing=json.loads((HERE/'replay_benchmark.json').read_text())
 assert timing['replays']==12 and timing['status']=='PASS'
 lines+=['','## Replay and measured inference','', 'All 12 final checkpoints replayed with zero observed local NLL and per-excerpt error. Model/optimizer/RNG/schedule/state are saved. No training occurs during replay.','', '| Arm | Mean of seed median step μs | Mean of seed p95 μs | Retained tensor bytes |','|---|---:|---:|---:|']
 for arm in ARMS:
  rr=[r for r in timing['rows'] if r['arm']==arm]
  lines.append(f"| {arm} | {statistics.mean(r['step_median_us'] for r in rr):.2f} | {statistics.mean(r['step_p95_us'] for r in rr):.2f} | {rr[0]['retained_tensor_bytes']} |")
 lines+=['','Timings are isolated single-thread CPU eager supplied-token steps after training exited, with warmup; they exclude sampling and are not mobile-device or optimized-kernel measurements. The immediate adapter still allocates its trace; a specialized deployment could omit that state.','',
 '## Next decision','',
 'Do not promote this as a strong language model or scale compute blindly. First test whether CSM can overfit a small real training excerpt beyond a count baseline, then isolate hard routing, admission, and the readout bottleneck using matched ablations. Check gradient magnitudes and routing margins against prediction diversity. Keep real-data quality, semantic record use, and state-preservation tests separate.']
 (HERE/'README.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
