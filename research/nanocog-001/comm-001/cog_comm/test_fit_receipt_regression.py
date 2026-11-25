from dataclasses import asdict
import pytest
from cog_comm.fit import Plan, derive_tokenizer, encode, merge_fit_counters, metadata, order_for


def test_actual_fit_metadata_receipt_joins_duplicate_train_families():
    families=[((f'pen number {i}','cup number '+str(i)),('blue','gold')) for i in range(8)]
    tok=derive_tokenizer(families)
    rows=encode(families,tok,512)
    plan=Plan('A','learned',171,steps=5)
    pre=metadata(rows,plan,order_for(len(rows),plan),tok)
    counters={'train_families':len(rows),'committed_updates':5,'answer_targets':40}
    with pytest.raises(TypeError,match='train_families'):
        dict(plan=asdict(plan),**pre,**counters,common_initial_sha256='inert')
    result=dict(plan=asdict(plan),**merge_fit_counters(pre,counters),common_initial_sha256='inert')
    assert set(result)==pre.keys()|counters.keys()|{'plan','common_initial_sha256'}
    assert result['train_families']==8 and result['committed_updates']==5
    assert result['answer_targets']==result['planned_answer_targets']==40
    for bad in [7,8.0]:
        with pytest.raises(ValueError,match='planned and observed counter disagree: train_families'):
            merge_fit_counters(pre,{**counters,'train_families':bad})
