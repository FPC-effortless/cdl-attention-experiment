"""Independent arithmetic + frozen-weight replay of 097e test decisions."""
import argparse, json, math, random, hashlib
from pathlib import Path
import torch
from experiment import corpus,Reader,RawPolicy,score_policy,score_outcomes,sha

def audit_seed(seed,folder):
    f=Path(folder)/f'result-{seed}.json'; r=json.loads(f.read_text());data,_=corpus()
    assert r['fresh_test_no_feedback']
    assert r['data_counts']=={k:len(v) for k,v in data.items()}
    q=list(data['test_known'])+list(data['test_unknown'])
    archive=list(data['test_known']);rng=random.Random(seed)
    dev_archive=list(data['dev_known']);rng.shuffle(dev_archive)
    rng.shuffle(archive)
    devreq=list(data['dev_known'])+list(data['dev_unknown']);rng.shuffle(devreq)
    rng.shuffle(q)
    assert r['test_input_sha']==sha('\n'.join(x['query'] for x in q))
    assert r['train_input_sha']==sha('\n'.join(x['body'] for x in data['train']))
    seen={};testkeys=set()
    for k in data:
        for z in data[k]:
            pair=(sha(z['query'].casefold()),sha(z['body']))
            assert pair[0] not in seen and pair[1] not in testkeys
            seen[pair[0]]=k;testkeys.add(pair[1])
    for method in ('char','word','learned'):
        mode='neural' if method=='learned' else method
        model=None
        if mode=='neural':
            state=torch.load(Path(folder)/f'097e-{seed}.pt',map_location='cpu',weights_only=True)
            assert state['seed']==seed
            model=RawPolicy(True);model.load_state_dict(state['state']);model.eval()
        predictions,times=score_policy(model,q,archive,mode)
        cutoff=r['calibration'][method]['threshold']
        replay=score_outcomes(predictions,q,archive,cutoff)
        saved=r['results'][method]
        for k in ('n','correct','wrong','accepted','abstained','unknown_false'):
            assert saved[k]==replay[k],(seed,method,k,saved[k],replay[k])
        assert saved['records']==replay['records'],(seed,method,'case-level mismatch')
    # No test-time updates: all learned checkpoint gradients were disabled at scoring.
    return {'seed':seed,'rows_verified':len(q),'methods_replayed':3,
            'feedback_actions':r['event_receipt']['actions'],
            'hash_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder',default='.');ap.add_argument('--seeds',nargs='*',type=int,default=[1701,1707,1711]);a=ap.parse_args()
    reports=[audit_seed(seed,a.folder) for seed in a.seeds]
    out=Path(a.folder)/'AUDIT.json';out.write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
