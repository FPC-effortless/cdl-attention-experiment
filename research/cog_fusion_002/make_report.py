"""Write the research report from completed, independently reconciled evidence."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

from common import HERE, PARENT, SEEDS, ARMS


def table(headers,rows):
    output=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    output += ['| '+' | '.join(map(str,row))+' |' for row in rows]
    return '\n'.join(output)


def read(name):
    return json.loads((HERE/'evidence'/name).read_text())


def yes(value):
    return 'PASS' if value else 'FAIL'


def ratio(row,num,den):
    return f"{row[num]}/{row[den]} ({100*row[num]/row[den]:.1f}%)"


def main():
    q=read('QUALIFICATION.json');language=read('LANGUAGE_PRESERVATION.json');runtime=read('RUNTIME_RESULTS.json')
    training=read('TRAINING_AUDIT.json')
    runs=q['runs'];decisions=q['decisions'];effect=q['paired_regime_effect']
    outcome=('The targeted familiar-regime repair passed all three seeds.'
             if decisions['targeted_familiar_repair_qualified'] else
             'The targeted familiar-regime repair did not pass every declared gate.')
    causal=('The matched comparison supports a specific benefit from carrying state during training.'
            if decisions['specific_regime_advantage_qualified'] else
            'The matched comparison did not qualify a specific advantage from carrying state during training.')
    text=['# COG-FUSION-002 — learning across generated answers','23 November 2026','## Decision',
          f'**{outcome}** {causal} New-wording qualification, old-regression preservation and original-full language recovery are reported separately below.',
          'This cycle retains COG-FUSION-001 unchanged. It tests a fixed training-regime repair in the same model and does not establish an architecture-wide result or a quality-matched efficiency advantage over attention.']
    labels={
        'all_continuous_training_adequacy':'Continuous arms: TRAIN competence',
        'all_continuous_familiar_repair':'Continuous arms: familiar multi-turn repair',
        'specific_regime_advantage_qualified':'Specific advantage over matched reset training',
        'all_continuous_novel_wording_absolute_gates':'Continuous arms: new-wording absolute gates',
        'all_continuous_old_regressions':'Earlier familiar/fresh/long regressions',
        'all_continuous_paired_parent_language_preservation':'Language preservation versus matching 001 parent',
        'all_continuous_original_full_language_recovery':'Language within 10% NLL of original full model',
        'targeted_familiar_repair_qualified':'Combined targeted familiar repair',
        'broader_wording_repair_qualified':'Combined repair including new wording',
    }
    text += [table(['Prespecified decision','Result'],[(label,yes(decisions[key])) for key,label in labels.items()]),
             '## What changed and what the comparison isolates',
             'The unchanged model stores 1,077,834 parameters, with 997,674 active and 80,160 retained native-board parameters disabled. Its neural continuation remains 10,592 bytes per world. The protected factual ledger and receipts can grow. No new trainable module, tokenizer, answer choice list or output mask was introduced.',
             'Each of the three matched pairs starts from its corresponding 001 memory endpoint. The added TRAIN corpus has 2,048 paired histories. Each history has eight observations and three questions, with actual generated answers consumed before later observations. Training includes the familiar wording bank plus new TRAIN paraphrases; both the old 001 held templates and new 002 held templates are excluded.',
             'Both arms receive the same family order, typed supervision, answer targets, language-rehearsal windows, optimizer and 1,200 updates. The continuous arm carries its actual generated continuation. The reset-training control computes each answer and then resets neural state, retaining the factual ledger. Both are evaluated continuously on held panels. A separate teacher-forced branch computes the character loss; its state is discarded.',
             'The first 100 updates use supervised typed execution and the remaining 1,100 use predicted writes and queries. Actual greedy-generated characters determine continuation throughout. Later losses can differentiate through the recurrent processing of those selected characters; the argmax token choices themselves are discrete. Rehearsal uses four 128-target TRAIN language windows every update with weight 0.5.',
             'Improvement over the frozen parent combines additional training, wording coverage, rehearsal and state policy. Only the matched continuous-versus-reset comparison isolates the training continuation policy. Equal examples and targets do not imply identical total work: generated lengths and the surviving gradient graph can differ.']
    main_rows=[]
    for label,arm,num,den in (
        ('Frozen parent: exact answers','parent','answers_exact','answers_total'),
        ('Reset-trained: exact answers','reset_training','answers_exact','answers_total'),
        ('Continuous-trained: exact answers','continuous','answers_exact','answers_total'),
        ('Frozen parent: complete pairs','parent','whole_session_pairs_exact','families'),
        ('Reset-trained: complete pairs','reset_training','whole_session_pairs_exact','families'),
        ('Continuous-trained: complete pairs','continuous','whole_session_pairs_exact','families')):
        main_rows.append([label,*[ratio(runs[f'{arm}_seed{s}']['panels']['test'],num,den) for s in SEEDS]])
    text += ['## Familiar multi-turn performance',table(['Measure','Seed17','Seed29','Seed41'],main_rows),
             'Each familiar panel has 128 paired families, 256 worlds, 768 generated answers and 2,048 proposed observations. A complete world requires every operation, every postobservation ledger, every query interpretation, every selected support identity and value, and every generated answer to match the reference. A complete pair requires both worlds. Correct final strings cannot conceal an incorrect history.']
    low,high=effect['percentile_95_interval']
    text += [f"The continuous-minus-reset complete-pair gain is **{100*effect['mean_accuracy_gain']:.2f} percentage points**, with a paired-family 95% percentile-bootstrap interval of **[{100*low:.2f}, {100*high:.2f}] points**. The 20,000 resamples retain all three seed contrasts for each sampled family. Seeds reuse the same held histories and do not triple the independent test sample. The interval is conditional on these three starting endpoints.",
             table(['Seed','Continuous-minus-reset complete-pair gain'],[(str(s),f"{100*effect['seed_accuracy_gains'][str(s)]:.2f} points") for s in SEEDS])]
    adequacy_rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            entry=runs[f'{arm}_seed{seed}'];p=entry['panels']['train_probe']
            adequacy_rows.append([f'{arm}, seed{seed}',ratio(p,'meaningful_post_answer_operations_exact','meaningful_post_answer_operations_total'),
                ratio(p,'query_support_joint_exact','queries_total'),ratio(p,'answers_exact','answers_total'),
                ratio(p,'whole_worlds_exact','worlds'),yes(entry['training_adequacy']['passed'])])
    text += ['## Training and regime adequacy',table(['Endpoint','Meaningful later writes','Query + support','Answers','Complete worlds','Gate'],adequacy_rows),
             'The fixed first-128 TRAIN probe is scored with predicted execution and actual generation under each arm\'s assigned training policy. It is a competence check on the assigned training distribution, not a generalization result. The gate requires 99% meaningful postanswer operations, 99% joint query/support correctness, 95% exact answers and 90% complete worlds. The training audit separately reports actual family and probe exposure under sampling with replacement.',
             'A failed competence gate means this finite budget did not establish learning adequacy. It does not establish that the architecture cannot learn the behavior. Successful competence likewise does not guarantee generalization to new wording or schedules.']
    exposure_rows=[]
    for seed in SEEDS:
        replay=training['replays'][str(seed)]
        exposure_rows.append([seed,f"{replay['distinct_training_families_seen']}/2048",
                              f"{replay['train_probe_families_sampled']}/128",
                              f"{replay['train_probe_families_sampled_during_predicted_execution']}/128",
                              len(replay['train_probe_unsampled_indices'])])
    text += [table(['Matched seed','Distinct families sampled','Probe families sampled','Probe sampled after warmup','Probe families never sampled'],exposure_rows),
             'The entire 128-family probe remains in the declared adequacy gate, including families never drawn by that seed. Exposure counts describe the fixed panel; they do not select an easier subset. The same draw sequence is independently reconstructed for each matched pair.']
    transfer_rows=[]
    for panel,label in (('fresh','New wording'),('long','Longer neutral delay')):
        for arm in ('parent','reset_training','continuous'):
            transfer_rows.append([f'{label}, {arm}',*[ratio(runs[f'{arm}_seed{s}']['panels'][panel],'answers_exact','answers_total') for s in SEEDS]])
            transfer_rows.append([f'{label}, {arm}, complete pairs',*[ratio(runs[f'{arm}_seed{s}']['panels'][panel],'whole_session_pairs_exact','families') for s in SEEDS]])
    text += ['## Unfamiliar wording and neutral-delay extrapolation',table(['Condition','Seed17','Seed29','Seed41'],transfer_rows),
             'New wording changes both lexicon and syntax while retaining the same sixteen entity names. LONG adds 64 neutral observations across two answer boundaries. Neither panel tests a new entity vocabulary, arbitrary task structure, unlimited horizon or free-form dialogue.']
    old_rows=[]
    for panel,label in (('test','Old familiar'),('fresh','Old held wording'),('long','Old long')):
        for measure,key in (('answers','exact_answers'),('complete histories','strict_whole_history_pairs')):
            values=[]
            for seed in SEEDS:
                report=read(f'regression/continuous_seed{seed}/RESULT.json')['panels'][panel]
                denom=report['worlds'] if measure=='answers' else report['families']
                values.append(f"{report['modes']['strict'][key]}/{denom}")
            old_rows.append([f'{label}: {measure}',*values])
    text += ['## Earlier exposed regression panels',table(['Continuous endpoints','Seed17','Seed29','Seed41'],old_rows),
             'These panels were already inspected during 001 and remain labelled regression evidence. Frozen-parent results are reused byte-for-byte with provenance receipts; adapted endpoints are scored again. The unchanged legacy evaluator also reruns its evidence-removal and fixed-state interventions. Those diagnostics cannot replace failed primary gates.']
    full=language['references']['original_connected_cog']['mean_nll']
    local=language['references']['original_local_only']['mean_nll']
    lang_rows=[['Original full Connected COG',f'{full:.7f}',f'{math.exp(full):.5f}','reference'],
               ['Original local-only parent',f'{local:.7f}',f'{math.exp(local):.5f}',f'{100*(local/full-1):+.2f}%']]
    for arm in ('parent','reset_training','continuous'):
        for seed in SEEDS:
            if arm=='parent': nll=language['references']['paired_fused001_parents'][str(seed)]['metrics']['mean_nll']
            else: nll=runs[f'{arm}_seed{seed}']['language']['paired_parent_nondegradation']['mean_nll']
            lang_rows.append([f'{arm}, seed{seed}',f'{nll:.7f}',f'{math.exp(nll):.5f}',f'{100*(nll/full-1):+.2f}%'])
    text += ['## Ordinary-language preservation',table(['Model','NLL, nats/character','Character perplexity','NLL gap to original full'],lang_rows),
             'All models receive the same 8,192 unique targets in 64 nonoverlapping validation windows, resetting recurrence independently per window. These are the exposed 001 regression windows. The repair-preservation gate permits at most 5% greater NLL than the matching fused parent. The separate original-full recovery reference permits at most 10% greater NLL than the original full Connected COG. The disabled-board baseline never substitutes for that stronger reference.',
             f"The independent inclusive comparison found **{len(q['language_gate_arithmetic']['locked_scorer_discrepancies'])}** discrepancies with the locked scorer\'s flags. The arithmetic clarification was recorded before held scoring; raw flags and measurements are retained."]
    runtime_rows=[]
    for entry in runtime['runs']:
        old=entry['exposed_001_regression'];new=entry['new_schedule']
        runtime_rows.append([f"{entry['arm']}, seed{entry['seed']}",f"{old['learned_questions_correct']}/{old['learned_question_count']}",
                            f"{new['learned_questions_correct']}/{new['learned_question_count']}",yes(new['complete_pipeline_pass']),yes(new['token_accounting_correct'])])
    text += ['## Actual persistent sessions',table(['Endpoint','Exposed 001 questions','New-schedule questions','New whole pipeline','Token accounting'],runtime_rows),
             'The new scenario first asks after two, four and six observations, then exercises a bridge change, a fresh-process restart, a cancelled transfer, relevant pending intent, a learned correction, retraction, reinstatement, supporting-source erasure and a historical receipt replay. All language uses familiar templates. The full scenario and its first three questions have no semantic-history overlap with the 002 splits.',
             'Pending scope and source erasure are supplied runtime controls, not learned language successes. Expected labels only score completed actual Session calls. A replayed historical request deliberately returns its old receipt; a new request asks about the current state. Restart correctness preserves accepted state, including any interpretation error. This shifted-schedule scenario is descriptive and has no post-hoc performance threshold.']
    intervention_rows=[]
    for arm in ('parent','reset_training','continuous'):
        for seed in SEEDS:
            path=HERE/'evidence/state_interventions'/f'{arm}_seed{seed}.json'
            record=json.loads(path.read_text());summary=record['summary']['meaningful']
            intervention_rows.append([f'{arm}, seed{seed}',*[f"{summary['variants'][v]['operations_exact']}/{summary['cases']}"
                for v in ('natural','all_reset','native_layers_reset','native_hidden_reset','decoder_reset')]])
    text += ['## Preselected state interventions',table(['Meaningful next-observation operations','Natural','All state reset','Layer states reset','Hidden only reset','Decoder only reset'],intervention_rows),
             'This diagnostic uses the first 32 familiar TEST families, four per kind, at both immediate postanswer boundaries. The next text and predicted factual ledger stay fixed. Only neural continuation is changed, and the natural trajectory continues for later cases. These altered-state results do not count as candidate successes. Interpret key margins only for the meaningful-operation subset; HOLD has no required destination key.']
    suites=ET.parse(HERE/'evidence/final_engineering_tests.xml').getroot().findall('.//testsuite')
    test_counts={key:sum(int(suite.get(key,'0')) for suite in suites) for key in ('tests','failures','errors','skipped')}
    passed=test_counts['tests']-sum(test_counts[key] for key in ('failures','errors','skipped'))
    text += ['## Engineering validation',
             f"The final 002 suite recorded **{passed} passes, {test_counts['failures']} failures, {test_counts['errors']} errors and {test_counts['skipped']} skips**. The fixtures cover data separation, candidate/scorer boundaries, actual generated-state training, state resets, accounting, matched evaluation admission, bootstrap arithmetic, interrupted-attempt recovery and final optimizer/RNG audits. The two Torch JIT deprecation warnings are retained in the receipt. These engineering checks support execution integrity; learned capability is assessed by the measured model panels above."]
    fit_rows=[];cpu=0;physical=0
    for seed in SEEDS:
        for arm in ARMS:
            r=read(f'runs/{arm}_seed{seed}/RESULT.json');cpu+=r['cpu_seconds'];physical+=r['physical_token_positions']
            fit_rows.append([f'{arm}, seed{seed}',r['steps'],f"{r['cpu_seconds']:.1f}",
                             r['input_chars'],r['generated_chars'],r['rehearsal_targets'],r['physical_token_positions'],r['distinct_training_families_seen']])
    text += ['## Work accounting and execution integrity',table(['Fit','Updates','Loop CPU s','Input chars','Generated chars','LM targets','Physical positions','Distinct families'],fit_rows),
             f'The six retained fitting loops used **{cpu:.1f} CPU seconds** and processed **{physical:,} physical token positions**, including padding and the distinct teacher-loss/actual-generation branches. Parallel CPU seconds are not elapsed wall time. The per-fit receipts retain initialization, batch/rehearsal draw, data/source, optimizer and endpoint hashes.']
    recovery=HERE/'evidence/RECOVERY_RESULT.json'
    if recovery.exists():
        value=json.loads(recovery.read_text())
        interrupted=q['operational_recovery']['interrupted_attempt_costs']
        lower=sum(r['recorded_cpu_seconds_lower_bound'] for r in interrupted)
        positions=sum(r['recorded_physical_token_positions'] for r in interrupted)
        text += [f'An operational recovery was required for **{len(value["recovered"])}** interrupted fit(s). The original exit statuses, partial traces and stdout remain archived. The retained endpoints restart from the exact original parent and command, without partial weights, extra updates or held-result selection. Discarded attempts add at least **{lower:.1f} reported training-loop CPU seconds** and **{positions:,} physical token positions**. The resulting measured lower bounds are **{cpu+lower:.1f} training-loop CPU seconds** and **{physical+positions:,} physical token positions**. Unreported interrupted work and startup are outside those lower bounds. Replacement whole-process CPU is retained separately and is not added again to the retained loop cost. A signal return code identifies termination, not its underlying cause.']
    text += ['The training audit independently replays all family and language-window draws, recomputes counters and coverage, checks optimizer/RNG state, and verifies unchanged inactive-board tensors. Independent evidence reconciliation reconstructs candidate ledgers and supporting reads from raw proposals, rather than trusting aggregate percentages.',
             'No quality-matched attention, GRU or alternative general sequence-model campaign was run. Unchanged parameters and inference code do not by themselves establish a latency or efficiency win. The user\'s inference-cost preference does not waive quality qualification.',
             '## Reproduction and remaining scope',
             'Use README.md and the companion archive\'s restoration guide. Source, final/optimizer checkpoints, exact data, raw traces, locked protocols, independent reviews and interrupted-attempt evidence are retained. The archive complements the pinned Git checkout rather than replacing its source files.',
             'The optional numerical component from 001 is unchanged. This cycle does not jointly train language to choose numerical operations or source addresses. Further research should follow the failed gates actually observed here, using a new protocol and held data rather than extending these endpoints after seeing their scores.']
    (HERE/'RESULT.md').write_text('\n\n'.join(text)+'\n')
    print(HERE/'RESULT.md')


if __name__=='__main__':
    main()
