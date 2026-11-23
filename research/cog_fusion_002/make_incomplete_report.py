"""Report the incomplete campaign without producing a capability verdict."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

from common import HERE, SEEDS, ARMS


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def main():
    evidence=HERE/'evidence'
    audit=read(evidence/'INCOMPLETE_CAMPAIGN_AUDIT.json')
    if audit['campaign_complete'] or audit['qualification_status']!='not_evaluated_incomplete_campaign':
        raise ValueError('this report is only for the explicitly incomplete campaign')
    fits={(row['arm'],row['seed']):row for row in audit['runs']}
    attempts=audit['incomplete_attempts']
    retry=next(row for row in attempts if row['attempt']=='permitted_retry')
    original=next(row for row in attempts if row['attempt']!='permitted_retry')
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            row=fits.get((arm,seed))
            rows.append([arm,seed,row['steps'] if row else retry['observed_updates'],
                         'available' if row else 'absent',
                         ('PASS' if row['individual_integrity_pass'] else 'FAIL') if row else 'not eligible'])
    cpu=sum(row['cpu_seconds'] for row in fits.values())
    positions=sum(row['counters']['physical_token_positions'] for row in fits.values())
    additional_cpu=sum(row['cpu_seconds_lower_bound'] for row in attempts)
    additional_positions=sum(row['counters_lower_bound'].get('physical_token_positions',0) for row in attempts)
    body=['# COG-FUSION-002 — incomplete training campaign','23 November 2026',
          '## Decision',
          '**The implementation is built, but the planned research comparison is incomplete. Five of six fits reached the fixed 1,200-update endpoint. Continuous seed 17 has no completed endpoint. No trained-model held evaluation has been performed, and the repair hypothesis remains untested by this cycle.**',
          f"The original continuous-seed-17 worker terminated with signal 9; its retained trace contains {original['observed_updates']} updates. When the permitted replacement's execution handle could not be recovered, its observed trace contained {retry['observed_updates']} updates. There is no replacement exit receipt, final checkpoint or recovery-completion receipt. The available execution tools cannot establish its terminal process status or failure cause. A missing control handle and unchanged files do not establish process death, an out-of-memory event or a timeout.",
          'The once-only recovery rule is preserved. No additional replacement was launched, no held performance was used to select an endpoint, and the five available fits are not substituted for the planned three-seed comparison. This is an execution limitation; it is not evidence that the model can or cannot learn the repair.',
          'Sources: [fixed protocol](PROTOCOL.md), [original campaign exits](evidence/CAMPAIGN_EXIT.json), [recovery declaration](evidence/RECOVERY_PROTOCOL.json), [recovery-start receipt](evidence/RECOVERY_STARTED.json), [execution-handle observation](evidence/EXECUTION_HANDLE_OBSERVATION.json), [independent disposition](review/INTERRUPTION_DISPOSITION_REVIEW.md).',
          '## Retained endpoints',table(['Arm','Seed','Recorded updates / 1,200','Final and optimizer','Individual integrity'],rows),
          'The separate [incomplete-campaign audit](evidence/INCOMPLETE_CAMPAIGN_AUDIT.json) checks individual completed fits without relaxing the original six-endpoint admission rule. It does not call a model, evaluate a TRAIN probe, read held scoring files or issue a campaign qualification.',
          'For each completed fit, it reconstructs the seeded family and language-window draw sequence, checks all 1,200 finite contiguous rows and counters, verifies final/optimizer identities and the exact saved Python sampling state, requires complete active AdamW moments, and confirms that inactive native-board tensors remain identical to the corresponding parent. The saved Torch RNG identity is observational. The unresolved replacement remains in place; stable byte snapshots preserve the observed files without asserting termination. Generated training strings were not retained, so generation totals receive consistency checks rather than exact string replay.',
          '## What was built',
          'The model architecture and inference implementation are inherited unchanged from COG-FUSION-001: **997,674 active parameters, 1,077,834 stored parameters and 10,592 bytes of neural continuation per world**. The 80,160 native-board parameters remain inactive. The factual ledger and request receipts can grow. This cycle adds no trainable module, tokenizer, answer-choice list or output mask.',
          'The repair addresses a failure exposed in 001: later observations could be misinterpreted when the native backbone carried state from generated answers. The new training families contain eight observations and three questions, asked after observations three, five and eight. Both arms actually generate their answers, consuming every emitted character, including the period when emitted. Generation has the unchanged twelve-character cap; a period is not forced. A separate teacher-forced branch supplies character loss and discards its resulting state.',
          'The continuous arm carries that generated neural continuation into later observations. The reset-training control generates the answer and then resets all neural state while retaining the factual ledger. Both arms would be evaluated continuously on the held panels. This matched comparison is necessary to separate a specific effect of carry training from improvement due to additional training data, supervision and rehearsal.',
          'The first 100 updates use supervised typed execution, followed by 1,100 updates with predicted writes and queries. The fit uses AdamW, gradient clipping at 1, native learning rate 0.000075, interface rate 0.0005 and the declared cosine schedule. Every update also rehearses four 128-target TRAIN language windows. Supplied data, sampling settings, source hashes, seeds, initialization, update counts and thresholds were fixed before training. Generated token inputs depend on the model.',
          'The delivered source includes training, label-isolated scoring, continuous-session evaluation, language preservation, persistent-session checks, state interventions, independent audits and a result summarizer. The complete-campaign audit and evaluation launcher still refuse the current incomplete inventory.',
          '## Data and coverage',
          'The new corpus contains 2,048 paired TRAIN families, 64 DEV families, 128 familiar TEST families, 128 new-wording families and 32 longer-delay families. The fixed adequacy probe contains the first 128 TRAIN families. The corpus uses the unchanged sixteen entity names and two relation types. An independent semantic replay checked 2,400 families, 4,800 worlds, 42,496 observations and 14,400 answers before fitting; held wording and semantic histories are separated from TRAIN.',
          'A TRAIN-pool family can remain unsampled under the finite sampler. Consequently, the fixed probe is a training-distribution adequacy check rather than a pure memorization check. It remains completely unscored in this campaign. Coverage below describes completed fits only.']
    coverage=[]
    for seed in SEEDS:
        r=audit['replays'][str(seed)]
        probe_sampled=sum(count>0 for count in r['full_family_sample_counts'][:128])
        coverage.append([seed,', '.join(arm for arm in ARMS if (arm,seed) in fits),
                         f"{r['distinct_training_families_seen']}/2048",
                         f"{probe_sampled}/128",
                         128-probe_sampled])
    body += [table(['Seed','Completed arms','Distinct TRAIN families drawn','Probe families drawn','Probe families never drawn'],coverage),
             'Sources: [data admission](review/DATA_ADMISSION.json), [pretraining review](evidence/PRETRAIN_REVIEW.json), and the independent completed-fit audit. The corpus and split files are retained in the companion evidence archive, and the audit retains the exact draw records.',
             '## What has not been tested',
             table(['Planned question','Status'],[
                 ['Fixed TRAIN-distribution adequacy gates','Not evaluated'],
                 ['Familiar multi-turn repair versus the frozen parents','Not evaluated'],
                 ['Specific advantage of continuous versus reset training','Not evaluated'],
                 ['New wording and longer neutral delays','Not evaluated'],
                 ['Earlier exposed regression preservation','Not evaluated'],
                 ['Ordinary-language preservation and original-full recovery','Not evaluated'],
                 ['New persistent-session schedule and learned postanswer parsing','Not evaluated'],
                 ['Preselected neural-state interventions','Not evaluated']]),
             'There is no QUALIFICATION.json, family-bootstrap result or held-performance figure. Low training loss and correct training-batch answers do not establish any of these gates. Existing 001 results retain their earlier interpretation; this incomplete cycle does not update them.',
             '## Engineering validation']
    receipt=evidence/'final_incomplete_engineering_tests.xml'
    if not receipt.exists(): receipt=evidence/'final_engineering_tests.xml'
    suites=ET.parse(receipt).getroot().findall('.//testsuite')
    counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    passed=counts['tests']-sum(counts[k] for k in ('failures','errors','skipped'))
    body += [f"The retained engineering suite records **{passed} passes, {counts['failures']} failures, {counts['errors']} errors and {counts['skipped']} skips**. The fixtures cover data separation, candidate/scorer boundaries, actual generated-state training, resets, accounting, matched evaluation admission, bootstrap arithmetic, one-attempt recovery and optimizer/RNG audits. These checks validate the machinery. The independently inspected real checkpoints support the narrower completed-fit integrity claims above.",
             '## Work accounting',
             table(['Completed fit','Training-loop CPU seconds','Physical token positions'],[
                 [f"{r['arm']}, seed {r['seed']}",f"{r['cpu_seconds']:.1f}",f"{r['counters']['physical_token_positions']:,}"]
                 for r in audit['runs']]),
             table(['Incomplete attempt','Recorded updates','Process status','Recorded CPU lower bound','Physical-position lower bound'],[
                 [r['attempt'],r['observed_updates'],str(r['process_status']),f"{r['cpu_seconds_lower_bound']:.1f}",f"{r['counters_lower_bound'].get('physical_token_positions',0):,}"]
                 for r in attempts]),
             f'The five retained fitting loops used **{cpu:,.1f} CPU seconds** and **{positions:,} physical token positions**. The two incomplete traces add at least **{additional_cpu:,.1f} CPU seconds** and **{additional_positions:,} positions**. The combined recorded lower bounds are therefore **{cpu+additional_cpu:,.1f} CPU seconds ({(cpu+additional_cpu)/3600:.2f} CPU hours)** and **{positions+additional_positions:,} positions**.',
             'These are training-loop measurements. Startup, unreported interrupted work, calibration, audits, tests and other orchestration are outside those totals. Parallel CPU seconds are not elapsed wall time. Physical positions include padding and the distinct teacher-loss, actual-generation and language-rehearsal branches. The two partial runs restarted from the parent and their updates must not be added together as progress toward one endpoint.',
             '## Continuation conditions',
             'The current study remains incomplete under its original rule. A future operational amendment could be considered only after trustworthy process-status evidence resolves the outstanding execution, and a supervisor can retain a terminal receipt. It must be recorded before any additional fit or held scoring, use a bounded attempt policy, retain every prior attempt, preserve the exact parent/seed/data/source and 1,200-update scientific settings, charge all discarded work, and explicitly identify later results as obtained under the amendment.',
             'The [independent disposition](review/INTERRUPTION_DISPOSITION_REVIEW.md) is a concrete unexecuted proposal. No replacement completion record, exit code or failure cause should be synthesized. The original recovery marker and the all-six gate remain intact.',
             '## Reproduction and limits',
             'The pinned source checkout and complementary archive retain all five completed final/optimizer checkpoints, both incomplete traces, the data and language inputs, required parent checkpoints, locked protocols, engineering receipts, independent reviews and observed snapshots. Read README.md and the accompanying restoration guide. The incomplete-campaign integrity audit can be repeated into fresh output and snapshot locations without model inference.',
             'The available checkpoints use controlled templates, fixed entities and explicit observation/question boundaries. There is no new generalization or ordinary-language claim from this cycle, no joint language-driven numerical computation result, and no quality-matched efficiency comparison against attention or another sequence model. A fixed neural state does not imply fixed total durable storage.']
    (HERE/'RESULT.md').write_text('\n\n'.join(body)+'\n')
    print(HERE/'RESULT.md')


if __name__=='__main__':
    main()
