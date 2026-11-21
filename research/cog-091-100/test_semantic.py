from semantic import make_data,to_tensor,Router,lex_first
import torch

def test_pairs_disjoint_and_novel_morphology():
    for seed in (9107,9203,9301):
        data=make_data(seed)
        pairs=[{tuple(z['pair']) for z in data[x]} for x in ('train','dev','test')]
        assert not(pairs[0]&pairs[1] or pairs[0]&pairs[2] or pairs[1]&pairs[2])
        assert len(data['test'])==60
        assert not ({r['text'] for r in data['train']} & {r['text'] for r in data['test']})
        assert sum(r['novelty']=='unseen_inflection' for r in data['test'])==30

def test_shapes_no_attention():
    data=make_data(9107);cache={}
    x,l,y=to_tensor(data['train'][:4],cache)
    for kind in ('bag','recurrent','parallel_specialists','self_attention'):
        model=Router(kind)
        assert tuple(model(x,l).shape)==(4,7)
        if kind != 'self_attention':
            assert not any('MultiheadAttention' in str(type(m)) for m in model.modules())

def test_frozen_embeddings_receive_no_gradient():
    m=Router('recurrent',True)
    x,l,y=to_tensor(make_data(9203)['train'][:4],{})
    torch.nn.functional.cross_entropy(m(x,l),y).backward()
    assert m.encoder.weight.grad is None
    assert m.gru.weight_ih_l0.grad is not None
