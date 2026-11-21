"""Non-oracle reward-channel, split, accounting, and replay tests."""
from pathlib import Path
import json
import torch
from experiment import TRAIN_MODULES,TEST_MODULES,DEV_MODULES,UNKNOWN_MODULES,make_corpus,sha,Encoder
from feedback import FRESH_MODULES,World,collect_feedback,train_feedback
from audit import audit
ROOT=Path(__file__).parent

def test_real_feedback_returns_only_bool():
    w=World(['visible unstructured document one','visible unstructured document two'])
    assert w.act(0,1) is False
    assert w.act(0,0) is True
    assert w.tries==2 and w.successes==1

def test_fresh_module_disjointness():
    assert not set(FRESH_MODULES)&(set(TRAIN_MODULES)|set(TEST_MODULES)|set(DEV_MODULES)|set(UNKNOWN_MODULES))
    tr=make_corpus(TRAIN_MODULES);te=make_corpus(FRESH_MODULES)
    assert not set(sha(x['body']) for x in tr)&set(sha(x['body']) for x in te)

def test_reward_counts_from_hidden_outcomes():
    tr=make_corpus(TRAIN_MODULES)
    events,counts=collect_feedback(tr)
    assert len(events)==counts['attempts']==591
    assert sum(int(x[2]) for x in events)==counts['successes']==196
    assert all(isinstance(x[0],str) and isinstance(x[1],str) and type(x[2]) is bool for x in events)

def test_feedback_optimizer_updates_weights():
    model=Encoder();initial=model.ff[0].weight.detach().clone()
    events=[('Which file is current?','An archived document',False),
            ('How to use a function?','A guide to function usage and examples',True),
            ('When to return?','Some unrelated discussion of imports and types',False),
            ('What is a test?','A description of testing behavior and results',True)]
    result,receipt=train_feedback(model,events,55,epochs=1)
    assert receipt['updates']==1
    assert not torch.equal(initial,result.ff[0].weight)

def test_replay_and_accounting():
    r=audit(replay=False)
    assert r['results']['d1']['self_supervised']['correct_total']==24
    assert r['results']['d2']['after_feedback']['correct_total']==35
    assert r['results']['d2']['char_tfidf']['correct_total']==216

def test_frozen_checkpoint_loads():
    for s in (1701,1707,1711):
        weights=torch.load(ROOT/f'feedback-{s}.pt',map_location='cpu',weights_only=True)
        model=Encoder();model.load_state_dict(weights['state']);model.eval()
        assert weights['seed']==s
