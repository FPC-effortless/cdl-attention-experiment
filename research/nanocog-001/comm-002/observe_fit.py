"""TrainLab-only admission-bound fit and observation receipt."""
import json
from pathlib import Path
import os
import sys
from hashlib import sha256

from cog_comm.role_fit import execute

def read(p):
    return json.loads(p.read_bytes())
def digest(p):
    return sha256(p.read_bytes()).hexdigest()

def main():
    recipe = read(Path('recipe.json'))
    c = read(Path('adequacy.json'))
    if recipe['research_adequacy']['sha256'] != digest(Path('adequacy.json')):
        raise ValueError("wrong contract")
    for p,h in c['inputs'].items():
        if digest(Path(p)) != h:
            raise ValueError("input changed")
    _,arm,train_sha = sys.argv
    out = Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    execute(Path('train.json'), train_sha, int(os.environ['TRAINLAB_SEED']), out, arm)
    count=read(out/'native-counters.json')['observations']
    trace=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
    fit=read(out/'fit.json')
    if fit['plan']['prefix_ce_weight'] != .25 or any(
        abs(x['objective'] - (.25*x['prefix_ce_nats_per_target'] + x['answer_ce_nats_per_target'] + x['paired_foil_softplus'])) > 3e-6
        for x in trace
    ):
        raise ValueError('role-002 objective not observed')
    nums = {'prefix_ce_weight':fit['plan']['prefix_ce_weight'],"observed_trace_steps":len(trace), "observed_trace_answer_targets":sum(x['answer_targets'] for x in trace),
            "observed_trace_prefix_targets":sum(x['prefix_targets'] for x in trace),
            "unique_train_at_128":len({x['family_ordinal'] for x in trace[:128]}),
            "endpoint_artifacts":sum(fit['endpoints'][str(t)]['checkpoint_sha256']==digest(out/f'checkpoint-{t}.pt') for t in (128,1280))}
    if sorted(x['step'] for x in trace) != list(range(1,1281)):
        raise ValueError("trace update schedule mismatch")
    inputs={**c['inputs'],'adequacy.json':digest(Path('adequacy.json'))}
    doc=dict(schema='trainlab.execution-adequacy-observations.v1',recipe_sha256=c['recipe_sha256'],
             inputs=inputs,producer={'file':'observe_fit.py','sha256':inputs['observe_fit.py']},
             observations={**count,**nums})
    with (out/'observations.json').open('x') as f:
        json.dump(doc,f,allow_nan=False,sort_keys=True)
if __name__=='__main__':
    main()
