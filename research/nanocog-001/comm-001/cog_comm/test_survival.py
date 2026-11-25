"""Synthetic inert wrapper checks, no staged TRAIN/CAL/DEV reads."""
import copy
import pytest
import torch
from .data import generate, ANSWERS
from .model import build_decoder
from .objective import Tokenizer
from .survival import locate,simulate,collect,summarize,project_counts


def fixture():
    fs=generate('train',8,1867,delays=(0,2))
    tok=Tokenizer.from_train([w.prompt for f in fs for w in f.worlds],ANSWERS)
    data=[]
    for i,f in enumerate(fs):
        for j,w in enumerate(f.worlds):
            ids,at,start,stop,value=locate(w.prompt,tok)
            assert value==w.answer and ids[at]==tok.index[value]
            assert 0<start<=at<stop<len(ids)-1
            if j:
                a,b=data[-1],(i,j,ids,at,start,stop)
                assert len(a[2])==len(b[2]) and a[3:]==b[3:]
            data.append((i,j,ids,at,start,stop))
    return tok,data


def test_manual_fifo_prequery_before_future_padding():
    w=[0,0,1,0,0,0,3,3,3,3]
    p=[True,True,False,True,False,True,True,True,False,True]
    before,after=simulate(w,p,6,2,3)
    assert before==[[5,3],[-1,-1],[-1,-1],[-1,-1]]
    assert after==[[-1,-1],[-1,-1],[-1,-1],[9,7]]


def test_projection_distinguishes_anchor_from_sentence_and_nested_stages():
    route=([True,True,False,True,True], [[1,-1],[0,-1],[-1,-1],[-1,-1]],
           [1,-1,-1,-1], [0,-1,-1,-1],[0,2],[1,3])
    a=project_counts(4,0,0,3,route)
    assert a['anchor']['emitted']==[0] and a['anchor']['retained']==[0]
    assert a['anchor']['learned_selected']==[] and a['anchor']['blind_selected']==[0]
    assert a['sentence_any']['learned_selected']==[1]
    assert a['sentence_any']['emitted']==[0,1]
    with pytest.raises(RuntimeError):
        bad=[True,True,False,True,True],route[1],[2,-1,-1,-1],*route[3:]
        project_counts(4,0,0,3,bad)


def test_native_c_full_trace_padded_single_parity_and_paired_counts():
    torch.set_num_threads(1)
    tok,data=fixture()
    # Two complete pairs, different prompt lengths; executes only toy fixtures.
    short_data=data[:4]
    assert len(set(len(e[2]) for e in short_data))==2
    m=build_decoder(tok.size,'C',width=32,layers=2,block=8,seed=171,retained_blocks=64).eval()
    batched=collect(short_data,m,171,batch_size=8)
    single=collect(short_data,m,171,batch_size=1)
    assert batched==single
    out=summarize(batched)
    assert out['seed=171/layer=0/view=anchor/anchor_age=all']['worlds']==4
    assert out['seed=171/layer=1/view=sentence_any/anchor_age=all']['families']==2
    assert all(-1<=a<x['delimiter_position'] for x in batched for a in x['learned_candidate_positions'])
    wrong=copy.deepcopy(batched)
    wrong[0]['world']=1
    with pytest.raises(ValueError):summarize(wrong)
