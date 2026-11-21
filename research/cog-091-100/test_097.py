"""Invariant tests for COG-ATTN-097a (no real actuator)."""
from experiment_097 import EpisodeIndex, make_store, split_queries, BUDGETS


def test_exact_index_conserves_all_six_records():
    rows,names=make_store(1536,9107);idx=EpisodeIndex(rows)
    for name in names:
        assert len(idx.index[name])==6
        assert len(set(idx.rows[i]['action'] for i in idx.index[name]))==6
        for action in range(6):
            ids,cost=idx.retrieve(name,[action],1)
            assert len(ids)==1 and idx.rows[ids[0]]['action']==action
            assert cost<=6


def test_missing_epoch_fails_closed_and_retains_archive():
    rows,names=make_store(1536,9301);idx=EpisodeIndex(rows);name=names[0]
    assert idx.retrieve(name,[0],1)[0]
    idx.invalidate(name)
    assert idx.retrieve(name,[0],1)[0]==()
    assert len(idx.index[name])==6
    assert idx.retrieve('missing',[0],1)[0]==()


def test_held_queries_have_no_training_utterance_overlap():
    from semantic import make_data
    for seed in (9107,9203,9301,9409,9503,9601):
        data=make_data(seed);train={x['text'] for x in data['train']}
        dev={x['text'] for x in data['dev']}
        held={x['text'] for x in data['test']}
        assert not train&held and not dev&held
        assert len(held)==60


def test_budget_monotonicity():
    rows,names=make_store(1536,9203);idx=EpisodeIndex(rows);name=names[1]
    ranks=[5,4,3,2,1,0]
    sets=[]
    for k in BUDGETS:
        ids,_=idx.retrieve(name,ranks,k)
        sets.append({idx.rows[i]['action'] for i in ids})
    assert all(sets[i]<=sets[i+1] for i in range(len(sets)-1))


def test_explicit_entity_routing_does_not_accept_unknown_refs():
    rows,names=make_store(1536,9301);idx=EpisodeIndex(rows)
    for fake in ('','wrong123','ticket9301x99999'):
        assert idx.retrieve(fake,[0,1,2],3)[0]==()
