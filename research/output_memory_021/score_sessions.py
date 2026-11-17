"""Independent rescoring of021 retained raw output tokens, no model execution."""
import json,random,re
from pathlib import Path

def decode(tokens):
    data=[]
    for t in tokens:
        if t==256:return bytes(data),True
        if not isinstance(t,int) or not 0<=t<256:return bytes(data),False
        data.append(t)
    return bytes(data),False

def score(records,results):
    expected={r['id']:r for r in records};seen=set();groups=[];sessions=[];exact_sessions=[];seconds=0.;answers=0;correct=0
    if len(expected)!=len(records):raise ValueError('duplicate roster ID')
    # Pair identity is global, not local to an execution batch. Otherwise four
    # distinct session IDs sharing one pair ID could be counted as two subjects.
    roster_pairs={}
    for record in records:
        variant=record.get('variant')
        if type(variant) is not int or variant not in (0,1):raise ValueError('invalid roster variant')
        roster_pairs.setdefault(record['pair_id'],[]).append(variant)
    if not roster_pairs or any(sorted(v)!=[0,1] for v in roster_pairs.values()):raise ValueError('global roster requires exactly two variants per pair')
    completed_pairs=set()
    configurations=set()
    for result in results:
        configurations.add(tuple(result[k] for k in ('kind','steps','condition','intervention','output_capacity')))
        if not result['rows'] or len(result['rows'])%2:raise ValueError('paired batch size')
        pairflags={};seconds+=result['seconds']
        for row in result['rows']:
            identity=row['id']
            if identity not in expected or identity in seen:raise ValueError('missing/duplicate/unknown ID')
            seen.add(identity);record=expected[identity]
            if row['pair_id']!=record['pair_id']:raise ValueError('pair ID')
            if type(row.get('variant')) is not int or row['variant']!=record['variant']:raise ValueError('result variant mismatch')
            if record['pair_id'] in completed_pairs:raise ValueError('pair repeated across batches')
            if len(row['answers'])!=len(record['answers']):raise ValueError('query roster')
            flags=[];exact=[]
            for index,(saved,gold) in enumerate(zip(row['answers'],record['answers'])):
                if saved['query_index']!=index or saved['identity']!=gold['identity'] or saved['payload_target']!=gold['target']:raise ValueError('label mismatch')
                raw=saved['raw_tokens']
                if len(raw)!=result['output_capacity']:raise ValueError('capacity mismatch')
                value,terminated=decode(raw);text=value.decode('utf8',errors='replace')
                matches=re.findall(r'(?<![A-Za-z0-9])([A-Z2-9]{3})(?![A-Za-z0-9])',text)
                payload='unknown' if text=='No code is currently assigned.' else matches[0] if len(matches)==1 else None
                good=terminated and payload==gold['target']
                canonical='No code is currently assigned.' if gold['target']=='unknown' else 'The current code is '+gold['target']+'.'
                exact_good=terminated and value==canonical.encode()
                if saved['correct']!=exact_good or saved['payload_correct']!=good:raise ValueError('claimed correctness mismatch')
                if saved['target']!=canonical or saved['prediction_bytes']!=list(value) or saved['prediction']!=text or saved['payload_prediction']!=payload:raise ValueError('rendered label/bytes mismatch')
                answers+=1;correct+=good;flags.append(good);exact.append(exact_good)
            sessions.append(all(flags));exact_sessions.append(all(exact));pairflags.setdefault(record['pair_id'],[]).append(flags[0])
        for pair_id,flags in pairflags.items():
            if len(flags)!=2:raise ValueError('split/incomplete pair')
            if pair_id in completed_pairs:raise ValueError('pair repeated across batches')
            completed_pairs.add(pair_id);groups.append(all(flags))
    if seen!=set(expected) or completed_pairs!=set(roster_pairs) or len(configurations)!=1:raise ValueError('incomplete or mixed results')
    rng=random.Random(21021)
    boot=sorted(sum(rng.choices(groups,k=len(groups)))/len(groups) for _ in range(2000))
    return {'sessions':len(sessions),'queries':answers,'payload_accuracy':correct/answers,'payload_session_accuracy':sum(sessions)/len(sessions),'exact_sentence_session_accuracy':sum(exact_sessions)/len(exact_sessions),'counterfactual_first_snapshot_pairs':len(groups),'both_snapshot_correct':sum(groups)/len(groups),'pair_bootstrap95':[boot[49],boot[1949]],'gate_pass':sum(sessions)/len(sessions)>=.95 and sum(exact_sessions)/len(exact_sessions)>=.95 and sum(groups)/len(groups)>=.95,'complete_batch_wall_seconds':seconds,'payload_metric':'literal-field presence only, not semantics; negation can contain target; exact sentence gate also mandatory','scope':'each axis/seed/config separately; finite constructed DEV; no claim of learned store competence from exact control'}
