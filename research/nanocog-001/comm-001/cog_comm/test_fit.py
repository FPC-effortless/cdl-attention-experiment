"""No models or optimizers are instantiated or run by these worker tests."""
from dataclasses import replace
import json

import pytest
import torch

from .fit import (ANSWERS, Plan, batch_inputs, checkpoint_packet, derive_tokenizer,
                  encode, metadata, order_for, read_train, read_vocab)
from .objective import SPECIAL, paired_loss


def fixtures(tmp_path):
    family_count=8
    prompts=[(f'The clerk asked Ada{i} a question about a train now.'+' Again.'*(i%4),
              f'The clerk asked Bea{i} a question about a train now.'+' Again.'*(i%4))
             for i in range(family_count)]
    source=tmp_path/'families.jsonl'
    rows=[dict(split='train',prompts=list(p),answers=['blue','gold']) for p in prompts]
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    train=read_train(source)
    tokenizer=derive_tokenizer(train)
    vocabulary=tmp_path/'vocabulary.json'
    vocabulary.write_text(json.dumps(dict(special=list(SPECIAL),words=list(tokenizer.words),
                                         minimum=5,answers=list(ANSWERS),shape=tokenizer.size)))
    return source,vocabulary,train,tokenizer


def test_exact_train_role_vocab_and_cap(tmp_path):
    data,vocab,rows,tok=fixtures(tmp_path)
    assert read_vocab(vocab,rows).words==tok.words
    data.write_text(data.read_text().replace('"train"','"dev"',1))
    with pytest.raises(ValueError,match='TRAIN-role'):
        read_train(data)
    v=json.loads(vocab.read_text());v["words"].append("unapproved")
    vocab.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='train-only'):
        read_vocab(vocab,rows)
    with pytest.raises(ValueError,match='over-cap'):
        encode(rows,tok,4)


def test_matched_schedule_and_dense_vs_unpadded_budget(tmp_path):
    _,_,rows,tok=fixtures(tmp_path);d=encode(rows,tok,512)
    base=Plan('A','learned',171,steps=5)
    signatures=set();seen=set()
    for name,mode in [('A','learned'),('B','learned'),('C','learned'),('C','content_blind')]:
        plan=replace(base,arm=name,route_mode=mode).validate()
        steps=order_for(len(d),plan)
        assert all(len(g)==4 and len(set(g))==4 for g in steps)
        actual=metadata(d,plan,steps,tok)
        signatures.add(actual['schedule_sha256']);seen.add(actual['planned_dense_input_tokens'])
        assert actual['planned_dense_input_tokens']==actual['planned_unpadded_tokens']+actual['planned_padding_tokens']
        assert actual['planned_unpadded_tokens']==actual['planned_prefix_targets']+actual['planned_answer_targets']
        assert actual['planned_answer_targets']==5*8
    assert len(signatures)==len(seen)==1


def test_padding_logits_and_answer_input_boundary(tmp_path):
    _,_,rows,tok=fixtures(tmp_path);d=encode(rows,tok,512)
    x,l,y,p=batch_inputs(d,(1,2,3,4))
    assert int(l.sum())==int(p.ne(-1).sum())+y.numel()
    for i,L in enumerate(l.tolist()):
        assert x[i,L-1].item()==5 and x[i,L:].eq(0).all()
        assert p[i,L-1:].eq(-1).all()
        assert torch.equal(p[i,:L-1],x[i,1:L])
        # Distinct TRAIN label-only colors are not appended and these simulated prompts do not mention them.
        assert y.reshape(-1)[i] not in x[i]
    # Dummy score arrays only, not a Decoder forward. Changing every synthetic pad score must be silent.
    z=torch.randn(*x.shape,tok.size)
    ordinary,tail=paired_loss(z,l,y,lm_targets=p)
    masked=z.clone()
    for i,L in enumerate(l.tolist()):masked[i,L:]=999
    new,after=paired_loss(masked,l,y,lm_targets=p)
    assert torch.equal(ordinary,new) and torch.equal(tail,after)


def test_ready_checkpoint_packet_primitive_weights_only(tmp_path):
    _,_,_,tok=fixtures(tmp_path)
    p=Plan('C','content_blind',371)
    path=tmp_path/'fake.pt'
    packet=checkpoint_packet({"weight":torch.arange(8,dtype=torch.float32)},p,tok,"a"*64,"b"*64,"c"*64,128)
    torch.save(packet,path)
    r=torch.load(path,weights_only=True,map_location="cpu")
    assert set(r)=={"format","plan","tokenizer_words","vocab_size","train_sha256","vocab_sha256",
                    "schedule_sha256","updates","state_dict"}
    assert r["plan"]["retained_blocks"]==64 and r["plan"]["route_mode"]=="content_blind"
    assert torch.equal(r["state_dict"]["weight"],packet["state_dict"]["weight"])
