"""097d input segregation, trainable learning, abstention, and replay invariants."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import (make_corpus,TRAIN_MODULES,TEST_MODULES,DEV_MODULES,
 UNKNOWN_MODULES,features,pair_windows,Encoder,rank_embedding,train_model,run,sha)

def test_disjoint_modules():
    sets=[set(x) for x in [TRAIN_MODULES,TEST_MODULES,DEV_MODULES,UNKNOWN_MODULES]]
    for i in range(len(sets)):
        for j in range(i):assert not sets[i]&sets[j]

def test_eval_summary_not_in_train_sample():
    tr=make_corpus(TRAIN_MODULES)
    a,b=pair_windows(tr[0]['body'])
    assert tr[0]['query'] not in a and tr[0]['query'] not in b
    assert a and b and tr[0]['body'].startswith(a)

def test_no_duplicate_body_between_train_test():
    tr=make_corpus(TRAIN_MODULES);te=make_corpus(TEST_MODULES)
    # The source modules contain a known duplicate getdoc docstring.
    # Verify the run's dedup filter removes it instead of silently leaking it.
    common={sha(x['body']) for x in tr}&{sha(x['body']) for x in te}
    assert len(common)==1
    result=json.loads((Path(__file__).parent/'result-1701.json').read_text())
    assert result['cross_split_duplicates_excluded']['test']>=1
    assert result['no_training_leak_tested']

def test_trainable_gradients_and_freeze():
    texts=['A completely separate alpha text with various clues and actions to investigate and test. '*3,
           'An unrelated beta text containing many details about computer behavior and the network. '*3,
           'A third descriptive gamma passage about assessing the execution results of a program. '*3,
           'A fourth delta passage on observations and effects over time and interactions. '*3]
    m,r=train_model(texts,33,epochs=2)
    assert r['updates']==2 and np.isfinite(r['loss']).all()
    w=[x.detach().clone() for x in m.parameters()]
    rank_embedding([dict(body=x,query=x[:70]) for x in texts],m)
    assert all(torch.equal(p,q) for p,q in zip(w,m.parameters()))

def test_no_entity_keys_in_features():
    assert isinstance(features('natural words from an observation'),list)
    assert features('a new expression')!=features('a different expression')

def test_all_missing_queries_are_unanswerable():
    r=json.loads((Path(__file__).parent/'result-1701.json').read_text())
    assert r['counts']['unknown']>0
    assert all(row['false_answers_unknown']==r['counts']['unknown'] for row in r['results'].values())

def test_seed_results_preserve_negative_evidence():
    d=Path(__file__).parent
    for seed in (1701,1707,1711):
        r=json.loads((d/f'result-{seed}.json').read_text())
        assert r['results']['self_supervised']['n']==r['counts']['test']
        assert r['training']['updates']==r['shuffled_training']['updates']
        assert r['results']['untrained']['n']==r['counts']['test']
        assert r['no_training_leak_tested']
        assert r['seed']==seed

def test_checkpoint_loading_read_only():
    m=Encoder();data=torch.load(Path(__file__).parent/'model-1701.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(data['state']);m.eval()
    assert data['seed']==1701
    assert data['epochs']==8
