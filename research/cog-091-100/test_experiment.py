from experiment import (make_history,index_raw,query_index,query_scan,
                        find_program,execute,PROGRAMS,memory_panel,skill_panel)

def test_lossless_retroactive_queries():
    rows=make_history(300,9107)
    ix,_=index_raw(rows)
    for row in rows:
        assert query_index(rows,ix,row['entity'],row['marker'])==row['day']
        assert query_scan(rows,row['entity'],row['marker'])==row['day']

def test_missing_and_ambiguous_index_fails_closed():
    rows=make_history(300,9203); ix,_=index_raw(rows)
    r=rows[0]
    assert query_index(rows,ix,r['entity'],'doesnotexist') is None
    rows.append(dict(entity=r['entity'],marker=r['marker'],day='WRONG',raw=r['raw']))
    ix,_=index_raw(rows)
    assert query_index(rows,ix,r['entity'],r['marker']) is None

def test_dsl_induction_is_bounded_and_has_no_oracle_labels():
    # Programs that cannot be uniquely identified from observations must abstain.
    program,matches=find_program([('abcdef','abcdef')])
    assert program is None and matches>1
    for p in PROGRAMS:
        assert isinstance(execute(p,'abcdef'),str)

def test_bad_witnesses_never_admitted():
    pred,matches=find_program([('abcdef','#wrong')])
    assert pred is None and matches==0

def test_replay_panels():
    r=memory_panel(300,9203,q_count=64)
    assert r['outcomes']['lossless_index']['correct']==64
    assert r['outcomes']['raw_scan']['correct']==64
    assert r['outcomes']['cached_scan']['correct']==64
    s=skill_panel(9107,episodes=20)
    assert s['corrupt_abstentions']==2
    assert s['wrong_actions']==0
