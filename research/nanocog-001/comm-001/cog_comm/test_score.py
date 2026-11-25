from .data import ANSWERS
from .score import score
import pytest


def test_counts_exact_pair_and_literal_slot_vs_later_route():
    ids={k:i for i,k in enumerate(ANSWERS)}
    p=[];t=[]
    for i in (0,1):
        common={"family_id":"F","world_index":i}
        p.append(dict(common,prompt_length=50,answer_scores=[float(j==i) for j in range(8)],prediction=i,
                      route_positions={"0":[4,7,9,-1]},no_route_prediction=0))
        t.append(dict(common,answer=ANSWERS[i],delay=2,support_start_token=6,support_end_token=9,
                      answer_source_token=8,age_eligible=True,local_suffix_equal=True))
    counts=score(p,t,ids)["all"]
    assert counts["both_exact"]==counts["both_literal_support_hits"]==1
    assert counts["both_color_position_hits"]==0 and counts["ordinary_success_lost_when_disabled"]==1
    with pytest.raises(ValueError,match="duplicated"):
        score([*p,p[0]],t,ids)
    p[0]["route_positions"]["0"][0]=49
    with pytest.raises(ValueError,match="impossible"):
        score(p,t,ids)
