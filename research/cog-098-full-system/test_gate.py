"""No mock model is allowed to satisfy the real qualification gate."""
from pathlib import Path
from gate import METHODS, methods_of, audit

def test_missing_agent_stays_missing(tmp_path):
    assert not methods_of(tmp_path/'agent.py')

def test_partial_agent_cannot_pass(tmp_path):
    p=tmp_path/'agent.py';p.write_text('class FullCOGAgent:\n def observe(self, x): pass\n')
    assert set(METHODS)-methods_of(p)

def test_no_fabricated_result_scores():
    r=audit()
    assert r['gate_type']=='source_and_receipt_readiness_not_model_quality'
    assert not any(s in r for s in ('accuracy','task_success','speedup'))

def test_no_authentic_receipts_means_not_qualified():
    r=audit()
    if not r['components']['authentic_holdout_receipts']:
        assert r['scientific_decision']=='NOT_QUALIFIED'
