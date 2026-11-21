"""No-oracle invariants and independent result arithmetic."""
import json,sys
from pathlib import Path
import pytest
from experiment import (make_corpus,MODULES,NEGATIVE_MODULES,BM25,Tfidf,LSA,tokens,sha,evaluate)
ROOT=Path(__file__).resolve().parent

def test_raw_corpus_real_authored_docs_without_slot_schema():
    c=make_corpus()
    assert len(c)>=150
    assert all(isinstance(x['body'],str) and len(x['body'])>=130 for x in c)
    assert all(isinstance(x['query'],str) for x in c)
    assert not any(k in c[0] for k in ('field','alias','gold','relation_id','target'))

def test_absent_source_modules_are_disjoint():
    a=make_corpus();b=make_corpus(NEGATIVE_MODULES)
    assert len(b)>=30
    assert set(x['source'] for x in a).isdisjoint(x['source'] for x in b)
    assert not set(sha(x['body']) for x in a) & set(sha(x['body']) for x in b)

def test_all_methods_search_global_corpus():
    docs=['one red apple on a table','migrations require durable records','quiet rivers cross a distant valley']
    q='durable migrations records'
    assert BM25(docs).rank(q)[0]==1
    assert Tfidf(docs,'word').rank(q)[0]==1
    assert Tfidf(docs,'char').rank(q)[0]==1
    assert LSA(docs,rank=1).rank(q)[0] in range(len(docs))

def test_gold_label_not_model_input():
    docs=['walk across bridge','network library token','file backup restore process']
    bm=BM25(docs);q='file restore';prediction=bm.rank(q)
    assert prediction[0]==2
    # Altering the private evaluator's source label does not alter the prediction.
    for hidden_gold in range(3):
        assert bm.rank(q)==prediction
        assert (prediction[0]==hidden_gold)==(hidden_gold==2)

def test_097b_cannot_ingest_schema_free_source():
    sys.path.insert(0,str(ROOT.parent/'cog-attn-091-100'))
    from experiment_097b import Archive
    with pytest.raises(KeyError,match='alias'):
        Archive([{'raw':'A naturally occurring observation, no pre-parsed identity or field.'}])

def test_results_scores_are_independently_reconstructible():
    r=json.loads((ROOT/'097c-results.json').read_text())
    assert r['n']==len(r['rows'])>=150
    assert r['n_unanswerable']==len(r['unanswerable'])>=30
    assert r['corpus_sha256']==sha('\n'.join(x['body'] for x in make_corpus()))
    for arm,stats in r['results'].items():
        rows=[x[arm] for x in r['rows']]
        absent=[x[arm] for x in r['unanswerable']]
        assert stats['correct']==sum(x['correct'] for x in rows)
        assert stats['n']==len(rows)
        assert stats['unanswerable_false_answers']==sum(x['false_answer'] for x in absent)
        assert stats['total_with_corpus_ingest_ms']>=stats['total_ms']

def test_no_execution_authority_in_stage():
    import experiment
    s=Path(experiment.__file__).read_text()
    assert 'issue(' not in s and '.commit(' not in s and 'Authority(' not in s
