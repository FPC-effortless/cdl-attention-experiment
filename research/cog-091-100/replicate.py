"""Fresh-seed synthetic replication with frozen methods and hyperparameters."""
import json,torch
from semantic import make_data,fit_evaluate,lex_first
from hypotheses import run_one
SEEDS=(9409,9503,9601)

def run():
    torch.set_num_threads(1)
    by_seed=[]
    for seed in SEEDS:
        d=make_data(seed)
        controls=dict(seed=seed,lexical_first=sum(lex_first(x['text'])==x['answer'] for x in d['test']),n=len(d['test']))
        models=[fit_evaluate(d,seed,mode,freeze) for mode,freeze in
                [('bag',False),('recurrent',True),('recurrent',False),('parallel_specialists',False),('self_attention',False)]]
        hyp=run_one(seed)
        by_seed.append(dict(seed=seed,controls=controls,models=models,hypotheses=hyp))
        print(seed,[(r['mode'],r['freeze_encoder'],r['correct']) for r in models],hyp['pair_top2'],'/',hyp['ambiguous_n'])
    return dict(name='Fresh split-seed replication, identical templates/hyperparameters',seeds=SEEDS,by_seed=by_seed,
                limitations=['Replicates synthetic generator only, not independent dataset or human language',
                             'Model training remains supervised by generated target actions'])
if __name__=='__main__':
    with open('replication-results.json','w') as f:json.dump(run(),f,sort_keys=True,indent=2)
