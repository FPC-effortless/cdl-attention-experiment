"""Matched continuous versus post-answer-reset training on actual generated turns.

The teacher-forced answer-loss branch is discarded. Only greedy-generated
characters (including their terminator) determine the next observation's state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from common import HERE, PARENT, sha256, write_json
import numpy as np
import torch
from torch.nn import functional as F
from fusion.data import digest
from fusion.engine import evidence_tensors, execute, greedy_generate, predicted_operation, strict_read
from fusion.model import FusionModel
from fusion.train import route_loss, tensor_digest


def operation_exact(predicted, expected):
    return (predicted[0] == expected[0] and (expected[0] == 0 or
            (predicted[1:3] == expected[1:3] and (expected[0] != 1 or predicted[3] == expected[3]))))


def answer_loss(model, state, resolutions, questions, families):
    prefix = '\nAnswer: '
    tokens, lengths = model.encode_texts([prefix + q['answer_text'][:-1] for q in questions])
    evidence, mask = evidence_tensors(resolutions)
    logits, discarded_state = model.decode(tokens, state, evidence, mask, lengths=lengths)
    del discarded_state
    labels = torch.full(tokens.shape, -100, dtype=torch.long)
    for i, question in enumerate(questions):
        ids = torch.tensor([model.stoi[c] for c in question['answer_text']])
        labels[i, len(prefix) - 1:len(prefix) - 1 + len(ids)] = ids
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    first = logits[:, len(prefix) - 1].log_softmax(-1)
    paired = []
    for i in range(len(families)):
        a, b = questions[2*i]['answer_text'][0], questions[2*i+1]['answer_text'][0]
        left, right = first[2*i], first[2*i+1]
        if a != b:
            ia, ib = model.stoi[a], model.stoi[b]
            paired.extend([F.relu(.5 - left[ia] + left[ib]), F.relu(.5 - right[ib] + right[ia])])
        else:
            paired.append(((left.exp() - right.exp()) * (left - right)).sum() / 2)
    return loss, torch.stack(paired).mean(), {
        'teacher_decode_chars': int(lengths.sum()), 'answer_targets': int((labels != -100).sum()),
        'teacher_decode_physical_positions': tokens.numel()}


def batch_loss(model, families, *, arm, teacher=False, lm_tokens=None, collect_states=False):
    if arm not in ('continuous', 'reset_training'):
        raise ValueError('unknown arm')
    worlds = [world for family in families for world in family['worlds']]
    schedule = [step['type'] for step in worlds[0]['steps']]
    if any([s['type'] for s in w['steps']] != schedule for w in worlds):
        raise ValueError('one batch requires a common explicit step schedule')
    batch = len(worlds)
    state = model.init_state(batch)
    ledgers = [{} for _ in worlds]
    observation_losses, query_losses, routing_losses, answer_losses, pair_losses = [], [], [], [], []
    metrics = {key: 0 for key in (
        'input_chars', 'input_physical_positions', 'teacher_decode_chars', 'teacher_decode_physical_positions',
        'answer_targets', 'generated_chars', 'generation_prefix_chars', 'rehearsal_targets',
        'rehearsal_physical_positions', 'write_exact', 'write_total', 'post_answer_write_exact',
        'post_answer_write_total', 'query_exact', 'query_total', 'generated_answer_exact',
        'generated_answer_total', 'answer_boundaries_reset', 'first_post_answer_write_exact',
        'first_post_answer_write_total', 'generation_decoder_calls', 'teacher_decoder_calls',
        'semantic_encoder_calls', 'rehearsal_calls')}
    seen_answer = False
    immediately_after_answer = False
    states = []
    for position, kind in enumerate(schedule):
        steps = [world['steps'][position] for world in worlds]
        tokens, lengths = model.encode_texts([step['text'] + '\n' for step in steps])
        metrics['input_chars'] += int(lengths.sum())
        metrics['input_physical_positions'] += tokens.numel()
        feature, semantics, state = model.encode_event(tokens, lengths, state)
        metrics['semantic_encoder_calls'] += 1
        if kind == 'observe':
            gold = {key: torch.tensor([step[key] for step in steps])
                    for key in ('operation', 'entity', 'relation', 'value')}
            loss = F.cross_entropy(semantics['operation'], gold['operation'])
            changed, assigned = gold['operation'] != 0, gold['operation'] == 1
            if bool(changed.any()):
                loss = loss + F.cross_entropy(semantics['entity'][changed], gold['entity'][changed])
                loss = loss + .5 * F.cross_entropy(semantics['relation'][changed], gold['relation'][changed])
            if bool(assigned.any()):
                loss = loss + F.cross_entropy(semantics['value'][assigned], gold['value'][assigned])
            observation_losses.append(loss)
            for i, step in enumerate(steps):
                predicted = predicted_operation(semantics, i)
                expected = tuple(step[key] for key in ('operation', 'entity', 'relation', 'value'))
                exact = int(operation_exact(predicted, expected))
                metrics['write_exact'] += exact
                metrics['write_total'] += 1
                if seen_answer:
                    metrics['post_answer_write_exact'] += exact
                    metrics['post_answer_write_total'] += 1
                if immediately_after_answer:
                    metrics['first_post_answer_write_exact'] += exact
                    metrics['first_post_answer_write_total'] += 1
                execute(ledgers[i], expected if teacher else predicted)
            immediately_after_answer = False
            continue
        if kind != 'ask':
            raise ValueError('unsupported step type')
        entities_gold = torch.tensor([q['query_entity'] for q in steps])
        plans_gold = torch.tensor([q['query_plan'] for q in steps])
        query_losses.append(F.cross_entropy(semantics['query_entity'], entities_gold)
                            + .5 * F.cross_entropy(semantics['query_plan'], plans_gold))
        entities_pred = semantics['query_entity'].argmax(-1)
        plans_pred = semantics['query_plan'].argmax(-1)
        metrics['query_exact'] += int(((entities_pred == entities_gold) & (plans_pred == plans_gold)).sum())
        metrics['query_total'] += batch
        entities = entities_gold if teacher else entities_pred
        plans = plans_gold if teacher else plans_pred
        resolutions = [strict_read(ledger, int(entity), int(plan))
                       for ledger, entity, plan in zip(ledgers, entities, plans)]
        routing_losses.append(route_loss(model, feature, steps, resolutions, teacher))
        answer, paired, accounting = answer_loss(model, state, resolutions, steps, families)
        metrics['teacher_decoder_calls'] += 1
        answer_losses.append(answer)
        pair_losses.append(paired)
        for key, count in accounting.items():
            metrics[key] += count
        # No no_grad or detach on a carried boundary: later semantic supervision
        # can differentiate through the backbone that consumed actual outputs.
        generated, generated_state = greedy_generate(model, state, resolutions)
        metrics['generated_chars'] += sum(map(len, generated))
        metrics['generation_prefix_chars'] += batch * len('\nAnswer: ')
        metrics['generation_decoder_calls'] += 1 + max(map(len, generated))
        metrics['generated_answer_exact'] += sum(text == q['answer_text'] for text, q in zip(generated, steps))
        metrics['generated_answer_total'] += batch
        if collect_states:
            states.append({'generated': generated, 'carried_state': generated_state})
        if arm == 'reset_training':
            state = model.init_state(batch)
            metrics['answer_boundaries_reset'] += batch
        else:
            state = generated_state
        seen_answer = True
        immediately_after_answer = True
    semantic = torch.stack(observation_losses).mean() + torch.stack(query_losses).mean()
    answer = torch.stack(answer_losses).mean()
    routing = torch.stack(routing_losses).mean()
    paired = torch.stack(pair_losses).mean()
    loss = semantic + .5 * routing + 2 * answer + .2 * paired
    rehearsal = torch.zeros(())
    if lm_tokens is not None:
        logits, _ = model.forward_lm(lm_tokens[:, :-1])
        rehearsal = F.cross_entropy(logits.flatten(0, 1), lm_tokens[:, 1:].flatten())
        loss = loss + .5 * rehearsal
        metrics['rehearsal_targets'] = lm_tokens[:, 1:].numel()
        metrics['rehearsal_physical_positions'] = lm_tokens[:, :-1].numel()
        metrics['rehearsal_calls'] = 1
    metrics.update(semantic_loss=float(semantic.detach()), answer_loss=float(answer.detach()),
                   route_loss=float(routing.detach()), pair_loss=float(paired.detach()),
                   lm_loss=float(rehearsal.detach()))
    metrics['physical_token_positions'] = (metrics['input_physical_positions'] + metrics['teacher_decode_physical_positions']
                                           + metrics['generation_prefix_chars'] + metrics['generated_chars']
                                           + metrics['rehearsal_physical_positions'])
    return (loss, metrics, states) if collect_states else (loss, metrics)


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(20090017 + args.seed)
    model = FusionModel.from_checkpoint(args.parent)
    payload = json.loads(args.train_data.read_text())
    families = payload['families'] if isinstance(payload, dict) else payload
    lm = np.memmap(args.shakespeare, dtype='<u2', mode='r')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError('new fit requires an empty output directory')
    dependencies = [HERE/'common.py', HERE/'data.py', HERE/'train.py', HERE/'PROTOCOL.md',
                    PARENT/'fusion/model.py', PARENT/'fusion/engine.py', PARENT/'fusion/data.py', PARENT/'fusion/train.py',
                    *sorted((PARENT/'fusion/native').glob('*.py'))]
    identity = {
        'study': 'COG-FUSION-002', 'arm': args.arm, 'seed': args.seed,
        'parent_sha256': sha256(args.parent), 'initial_tensor_sha256': tensor_digest(model),
        'parameters': model.parameter_summary(), 'train_data_sha256': sha256(args.train_data),
        'train_families_sha256': digest(families), 'shakespeare_train_sha256': sha256(args.shakespeare),
        'source_sha256': {str(p.relative_to(HERE.parent)): sha256(p) for p in dependencies},
        'steps': args.steps, 'teacher_steps': args.teacher_steps, 'batch_families': args.pairs,
        'optimizer': 'AdamW', 'weight_decay': .01, 'gradient_clip': 1.,
        'native_lr': args.native_lr, 'interface_lr': args.lr, 'lm_weight': .5,
        'lm_every_updates': 1, 'lm_windows_per_update': 4, 'lm_targets_per_window': 128,
        'rollout': 'actual greedy emitted chars incl terminator; separate teacher-loss branch discarded',
        'bptt': 'full differentiable generated-state carry' if args.arm == 'continuous' else 'all neural state reset after answers',
        'selection': 'fixed final update; held and DEV files unavailable to trainer',
        'threads': args.threads, 'cpu_cap': args.cpu_cap}
    write_json(args.output/'IDENTITY.json', identity)
    parameters = [p for name,p in model.named_parameters() if name.startswith('native.')]
    interfaces = [p for name,p in model.named_parameters() if not name.startswith('native.')]
    optimizer = torch.optim.AdamW([{'params':parameters,'lr':args.native_lr},
                                  {'params':interfaces,'lr':args.lr}],weight_decay=.01)
    trace = hashlib.sha256()
    coverage = set()
    counters = {}
    started, cpu_started = time.perf_counter(), time.process_time()
    model.train()
    for step in range(1,args.steps+1):
        indices=[rng.randrange(len(families)) for _ in range(args.pairs)]
        starts=[rng.randrange(len(lm)-129) for _ in range(4)]
        trace.update(json.dumps([indices,starts]).encode())
        coverage.update(indices)
        rehearsal=torch.tensor(np.stack([np.array(lm[start:start+129],dtype=np.int64) for start in starts]))
        for group,base in zip(optimizer.param_groups,(args.native_lr,args.lr)):
            group['lr']=base*(.2+.8*(1+math.cos(math.pi*(step-1)/args.steps))/2)
        optimizer.zero_grad(set_to_none=True)
        loss,metrics=batch_loss(model,[families[i] for i in indices],arm=args.arm,
                                teacher=step<=args.teacher_steps,lm_tokens=rehearsal)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('nonfinite training gradient')
        optimizer.step()
        for key,value in metrics.items():
            if isinstance(value,int):
                counters[key]=counters.get(key,0)+value
        row={'step':step,'loss':float(loss.detach()),**metrics,'gradient_norm':float(norm),
             'teacher_typed_execution':step<=args.teacher_steps,
             'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu_started}
        with (args.output/'train.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if step==1 or step%50==0 or step==args.steps:
            print(json.dumps(row),flush=True)
        if row['cpu_seconds']>args.cpu_cap:
            torch.save(model.checkpoint_payload(step=step,study='COG-FUSION-002',stopped='cpu_cap',**counters),args.output/'incomplete.pt')
            raise RuntimeError('prospective CPU cap exceeded; incomplete checkpoint retained')
    elapsed={'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu_started}
    model.eval()
    torch.save(model.checkpoint_payload(study='COG-FUSION-002',step=args.steps,seed=args.seed,arm=args.arm,
        parent_sha256=identity['parent_sha256'],batch_trace_sha256=trace.hexdigest(),**elapsed,**counters),args.output/'final.pt')
    torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                'python_rng':rng.getstate(),'step':args.steps},args.output/'optimizer.pt')
    result={**identity,**elapsed,**counters,'batch_trace_sha256':trace.hexdigest(),
            'distinct_training_families_seen':len(coverage),'training_families_available':len(families),
            'final_model_identity':model.model_identity(),'checkpoint_sha256':sha256(args.output/'final.pt'),
            'status':'completed'}
    write_json(args.output/'RESULT.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--train-data',type=Path,required=True)
    parser.add_argument('--shakespeare',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=('continuous','reset_training'),required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--teacher-steps',type=int,default=100)
    parser.add_argument('--pairs',type=int,default=4)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--cpu-cap',type=float,default=3600)
    parser.add_argument('--lr',type=float,default=.0005)
    parser.add_argument('--native-lr',type=float,default=.000075)
    run(parser.parse_args())
