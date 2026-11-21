"""No-ID retrieval, revision and untrained-domain negative control tests."""
from experiment_097b import Archive, make_archive, questions, make_name, FIELDS, build_witnesses, WitnessReader, query_string, relation_question


def fixture():
    rows,names=make_archive(1103,2048,234)
    return Archive(rows),names


def test_unseen_entity_not_supplied_as_structured_key():
    idx,names=fixture();s=query_string(names[155], FIELDS[1][1][0],'ellipsis')
    n,ids,matches=idx.candidates(s)
    assert n==names[155] and len(ids)==7 and matches==1


def test_ambiguity_and_missing_references_defer():
    idx,names=fixture()
    x=f'We discussed {names[150]} and {names[151]}; now which invoice?'
    assert idx.candidates(x)[1]==() and idx.candidates(x)[2]==2
    assert idx.candidates('Regarding nonexistent Neverthere which invoice?')[1]==()


def test_late_correction_preserves_old_raw_episode():
    idx,names=fixture();name=names[5]
    old=[r for r in idx.rows if r['alias']==name and r['field']==1]
    assert len(old)==2 and {r['version'] for r in old}=={1,2}
    assert idx.rows[idx.latest[(name,1)]]['version']==2
    answer,_=idx.direct_scan(name,1)
    assert answer['version']==2


def test_held_entity_disjoint_and_novel_field():
    _,names=fixture();tr=questions(1103,names[:108],'train');te=questions(1103,names[144:234],'test')
    assert not ({x['name'] for x in tr}&{x['name'] for x in te if x['name']})
    assert sum(x['kind']=='unseen_field' for x in te)==23
    assert not any(x['field']==6 for x in tr)


def test_shuffled_witness_breaks_link_without_changing_count():
    idx,names=fixture();e=build_witnesses(names[:108],idx)
    a=WitnessReader(idx.rows,e);b=WitnessReader(idx.rows,e,shuffle=True)
    assert len(a.supports)==len(b.supports)==14
    query=query_string(names[155],FIELDS[0][2][0],'direct')
    ids=idx.candidates(query)[1];docs=[idx.rows[i]['raw'] for i in ids]
    positive,_=a.propose(query,names[155],docs)
    wrong,_=b.propose(query,names[155],docs)
    assert positive is not None and wrong is not None and positive != wrong


def test_supported_phrase_is_not_zero_shot_semantics():
    idx,names=fixture();e=build_witnesses(names[:108],idx);w=WitnessReader(idx.rows,e)
    unknown='what was the ceremonial soft triangle of that request'
    candidate,score=w.propose(query_string(names[170],unknown,'direct'),names[170],[idx.rows[i]['raw'] for i in idx.candidates(query_string(names[170],unknown,'direct'))[1]])
    assert candidate is None
