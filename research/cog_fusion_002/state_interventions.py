"""Predeclared next-observation state interventions; never used for selection."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

from common import HERE, PARENT, sha256, write_json
import torch
from fusion.engine import execute, greedy_generate, predicted_operation, strict_read
from fusion.model import FusionModel, clone_state, state_map
from evaluate import operation_exact


def variant_state(model, state, variant, batch):
    if variant == 'natural':
        return state
    if variant == 'all_reset':
        return model.init_state(batch)
    altered = clone_state(state)
    if variant == 'native_layers_reset':
        altered['native']['layers'] = state_map(altered['native']['layers'],torch.zeros_like)
    elif variant == 'native_hidden_reset':
        for layer in altered['native']['layers']:
            layer['hidden'] = torch.zeros_like(layer['hidden'])
    elif variant == 'decoder_reset':
        altered['decoder'] = torch.zeros_like(altered['decoder'])
    else:
        raise ValueError('unregistered state intervention')
    return altered


@torch.inference_mode()
def run_panel(model,families,batch_families=4):
    variants=('natural','all_reset','native_layers_reset','native_hidden_reset','decoder_reset')
    rows=[]
    for start in range(0,len(families),batch_families):
        selected=families[start:start+batch_families]
        worlds=[world for family in selected for world in family['worlds']]
        state=model.init_state(len(worlds));ledgers=[{} for _ in worlds]
        history=[[] for _ in worlds];previous=None;question_number=0
        for position in range(len(worlds[0]['steps'])):
            steps=[world['steps'][position] for world in worlds]
            kind=steps[0]['type']
            tokens,lengths=model.encode_texts([step['text']+'\n' for step in steps])
            if kind=='observe' and previous=='ask':
                variants_output={}
                for variant in variants:
                    _,semantics,_=model.encode_event(tokens,lengths,variant_state(model,state,variant,len(worlds)))
                    variants_output[variant]=semantics
                for i,step in enumerate(steps):
                    expected=tuple(step[k] for k in ('operation','entity','relation','value'))
                    scores={}
                    for variant,semantics in variants_output.items():
                        predicted=predicted_operation(semantics,i)
                        logits=semantics['entity'][i].clone()
                        correct=logits[step['entity']].item();logits[step['entity']]=-torch.inf
                        scores[variant]={'predicted_operation':list(predicted),
                                         'operation_exact':operation_exact(predicted,expected),
                                         'correct_entity_logit_margin':correct-float(logits.max())}
                    row={'family_id':selected[i//2]['family_id'],'side':i%2,'kind':selected[i//2]['kind'],
                         'after_question':question_number,'step_position':position,'text':step['text'],
                         'meaningful':step['operation']!=0,'expected_operation':list(expected),
                         'preceding_generated_answers':list(history[i]),'variants':scores,
                         'decoder_reset_semantic_logits_bitwise_equal':all(torch.equal(variants_output['natural'][k][i],
                                     variants_output['decoder_reset'][k][i]) for k in variants_output['natural'])}
                    rows.append(row)
            _,semantics,state=model.encode_event(tokens,lengths,state)
            if kind=='observe':
                for i in range(len(worlds)):
                    execute(ledgers[i],predicted_operation(semantics,i))
            else:
                question_number+=1
                entities=semantics['query_entity'].argmax(-1).tolist()
                plans=semantics['query_plan'].argmax(-1).tolist()
                resolutions=[strict_read(ledger,e,p) for ledger,e,p in zip(ledgers,entities,plans)]
                generated,state=greedy_generate(model,state,resolutions)
                for record,text in zip(history,generated):
                    record.append(text)
            previous=kind
    summary={}
    for selection,chosen in (('all',rows),('meaningful',[row for row in rows if row['meaningful']])):
        summary[selection]={'cases':len(chosen),'variants':{
            variant:{'operations_exact':sum(r['variants'][variant]['operation_exact'] for r in chosen),
                     'mean_correct_entity_logit_margin':sum(r['variants'][variant]['correct_entity_logit_margin']
                                                           for r in chosen)/len(chosen) if chosen else None}
            for variant in variants}}
    summary['decoder_reset_logits_equal_cases']=sum(row['decoder_reset_semantic_logits_bitwise_equal'] for row in rows)
    return {'summary':summary,'rows':rows,'selection':'first 32 TEST families, four per lifecycle kind; no selection by model result',
            'scope':'same next-observation text; only neural state changed; actual predicted ledger persists; natural path continues'}


def run(args):
    if args.output.exists():
        raise FileExistsError('state-intervention output exists')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    families=json.loads(args.data.read_text())['families'][:32]
    model=FusionModel.from_checkpoint(args.checkpoint).eval()
    output=run_panel(model,families)
    output.update(schema='cog-fusion-002-state-interventions-v1',checkpoint_sha256=sha256(args.checkpoint),
                  data_sha256=sha256(args.data),source_sha256=sha256(Path(__file__)),
                  family_ids=[f['family_id'] for f in families])
    write_json(args.output,output)
    print(json.dumps({'output':str(args.output),'summary':output['summary']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=HERE/'evidence/data/test.json')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
