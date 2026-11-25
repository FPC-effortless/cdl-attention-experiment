import json
import torch
import pytest
from .data import ANSWERS
from .model import build_decoder
from .objective import Tokenizer
from .predict import predict_rows,read_actor


def test_predict_refuses_targets_and_trace_has_no_future_slot(tmp_path):
    text=("The badge is blue. What color is the badge?",
          "The badge is gold. What color is the badge?")
    a=[{"family_id":"inert","world_index":j,"prompt":p} for j,p in enumerate(text)]
    def save():
        f=tmp_path/"actor.jsonl"
        f.write_text(''.join(json.dumps(r)+'\n' for r in a))
        return f
    t=Tokenizer.from_train(text,ANSWERS,minimum=1)
    torch.set_num_threads(1)
    p=predict_rows(build_decoder(t.size,"C").eval(),t,read_actor(save()))
    assert len(p)==2 and all(len(r['answer_scores'])==8 for r in p)
    for r in p:
        assert all(len(v)==4 and all(-1<=i<r["prompt_length"]-1 for i in v)
                   for v in r["route_positions"].values())
        assert len(r["no_route_scores"])==8
    a[0]["answer"]="blue"
    with pytest.raises(ValueError,match="raw-text"):
        read_actor(save())
