"""Independent adversarial semantic checks; constructed data only.

The legacy fork witness reconstructs the previously inspected revision-only
comparison rule. It is not a claim that the old complete source was executed.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference" / "semantic_kernel.py"
spec = importlib.util.spec_from_file_location("reviewed_semantic_kernel", SOURCE)
k = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = k
spec.loader.exec_module(k)

A=("scope","alice","holder")
B=("scope","bob","holder")
C=("scope","carol","holder")
D=("scope","dave","holder")
E=("scope","eve","holder")
outcomes=[]

def put(s,*ops,**kwargs):
    r=k.commit(s,k.prepare(s,ops),**kwargs)
    assert r.accepted, r.status
    return r.store

def case(name,fn):
    try:
        detail=fn()
        if detail is False:
            raise AssertionError("predicate is false")
        outcomes.append({"name":name,"status":"PASS","detail":detail})
    except Exception as exc:
        outcomes.append({"name":name,"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)})

def rejected_constructor(make):
    try:
        make()
    except (ValueError,TypeError):
        return True
    return False

empty=k.Store(capacity=8)
base=put(empty,k.Operation("set",A,value="red"),k.Operation("set",B,value="blue"))
green=put(base,k.Operation("set",A,value="green"))
blue=put(base,k.Operation("set",A,value="blue"))

def legacy_fork():
    legacy_dependencies=((A,green.revision(A)),)
    old_rule_accepts=all(blue.revision(key)==rev for key,rev in legacy_dependencies)
    assert old_rule_accepts and k.resolve(green,A).value!=k.resolve(blue,A).value
    return {"old_rule_would_accept":old_rule_accepts,"same_revision":green.revision(A),
            "cached_branch_value":k.resolve(green,A).value,
            "actual_other_branch_value":k.resolve(blue,A).value,
            "execution_scope":"reconstructed old revision-only predicate; valid current API states"}
case("legacy_revision_only_fork_counterexample",legacy_fork)

def fork_cache():
    _,cache,_=k.cached_resolution(green,A)
    answer,_,hit=k.cached_resolution(blue,A,cache=cache)
    assert not hit and answer.value=="blue"
    return {"hit":hit,"answer":answer.value}
case("divergent_valid_forks_cannot_share_incompatible_cache",fork_cache)

linked=put(base,k.Operation("link",C,source=A),k.Operation("link",D,source=C))
lg=put(linked,k.Operation("set",A,value="green"))
lb=put(linked,k.Operation("set",A,value="blue"))

def transitive_copy():
    tx=k.prepare(lg,[k.Operation("copy",E,source=D)])
    r=k.commit(lb,tx)
    assert not r.accepted and r.store is lb and lb.cell(E) is None
    return {"status":r.status,"read_keys":[str(x[0]) for x in tx.expected]}
case("transitive_copy_rejects_equal_counter_different_value_fork",transitive_copy)

def observed_fork():
    tx=k.prepare(green,[k.Operation("set",C,value="derived")],observed=[A])
    r=k.commit(blue,tx)
    assert not r.accepted and r.store is blue
    return r.status
case("predicate_read_rejects_equal_counter_different_value_fork",observed_fork)

def batch_snapshot():
    s=put(base,k.Operation("set",A,value="new"),k.Operation("copy",C,source=A))
    assert k.resolve(s,A).value=="new" and k.resolve(s,C).value=="red"
    return {"written":k.resolve(s,A).value,"copied":k.resolve(s,C).value}
case("batch_copy_reads_original_snapshot_not_prior_operation",batch_snapshot)

def hop_cache():
    r,cache,_=k.cached_resolution(linked,D,max_links=1)
    assert r.status=="hop_limit"
    r,_,hit=k.cached_resolution(linked,D,max_links=2,cache=cache)
    assert not hit and r.value=="red"
    return {"larger_cap_hit":hit,"value":r.value}
case("hop_cap_is_part_of_cache_contract",hop_cache)

def aba():
    _,cache,_=k.cached_resolution(base,A)
    d=put(base,k.Operation("delete",A))
    restored=put(d,k.Operation("set",A,value="red"))
    r,_,hit=k.cached_resolution(restored,A,cache=cache)
    assert not hit and r.value=="red" and restored.revision(A)>base.revision(A)
    return {"old_revision":base.revision(A),"new_revision":restored.revision(A)}
case("delete_and_same_value_restore_invalidates_observation",aba)

def absence():
    tx=k.prepare(base,[k.Operation("set",A,value="derived")],observed=[C])
    inserted=put(base,k.Operation("set",C,value="arrived"))
    r=k.commit(inserted,tx)
    assert not r.accepted and r.store is inserted and k.resolve(r.store,A).value=="red"
    return r.status
case("absence_predicate_conflict_preserves_entire_batch",absence)

def revision_batch():
    s=k.Store(((A,k.Cell(1,k.Payload("value",value="red"),"typed-string-v1")),
               (B,k.Cell(k.MAX_REVISION,k.Payload("value",value="blue"),"typed-string-v1"))))
    tx=k.prepare(s,[k.Operation("set",A,value="new"),k.Operation("set",B,value="new")])
    r=k.commit(s,tx)
    assert not r.accepted and r.store is s and k.resolve(s,A).value=="red"
    return r.status
case("revision_exhaustion_on_one_member_rejects_whole_batch",revision_batch)

def capacity_batch():
    s=k.Store(base.entries,capacity=2)
    tx=k.prepare(s,[k.Operation("delete",A),k.Operation("set",C,value="new")])
    r=k.commit(s,tx)
    assert not r.accepted and r.store is s and k.resolve(s,A).value=="red"
    return {"status":r.status,"note":"tombstones do not silently free a key identity"}
case("capacity_failure_cannot_half_delete_or_silently_evict",capacity_batch)

case("mutable_inner_store_entry_rejected",
     lambda: rejected_constructor(lambda:k.Store(([A,base.cell(A)],))))
case("non_epoch_object_rejected",
     lambda: rejected_constructor(lambda:k.Store(epochs={"schema":"typed-string-v1"})))

def invalid_epoch_values():
    bad=[("authority",True),("authority",-1),("authority",[]),("stream",[]),
         ("policy",None),("schema",[]),("model",{}),("tokenizer",[]) ]
    accepted=[]
    for field,value in bad:
        if not rejected_constructor(lambda f=field,v=value:k.Store(epochs=dataclasses.replace(k.Epochs(),**{f:v}))):
            accepted.append(field+":"+repr(value))
    assert not accepted, "accepted malformed epoch values: "+str(accepted)
    return {"rejected_cases":len(bad)}
case("epoch_fields_have_immutable_typed_domains",invalid_epoch_values)

def tx_malformed_expected():
    tx=dataclasses.replace(k.prepare(base,[k.Operation("set",A,value="new")]),expected=((A,),))
    r=k.commit(base,tx)
    assert not r.accepted and r.store is base
    return r.status
case("malformed_expected_row_returns_rejection",tx_malformed_expected)

def tx_malformed_writes():
    tx=dataclasses.replace(k.prepare(base,[k.Operation("set",A,value="new")]),writes=((A,),))
    r=k.commit(base,tx)
    assert not r.accepted and r.store is base
    return r.status
case("malformed_write_row_returns_rejection",tx_malformed_writes)

def tx_mutable_row():
    valid=k.prepare(base,[k.Operation("set",A,value="new")])
    tx=dataclasses.replace(valid,expected=([valid.expected[0][0],valid.expected[0][1]],))
    r=k.commit(base,tx)
    assert not r.accepted and r.store is base
    return r.status
case("mutable_expected_row_rejected",tx_mutable_row)

def cap_validation(value):
    _,cache,_=k.cached_resolution(base,A,max_links=1)
    try:
        k.cached_resolution(base,A,max_links=value,cache=cache)
    except ValueError:
        return {"rejected":repr(value)}
    raise AssertionError("cache hit bypassed integer hop-cap validation")
case("cache_hit_does_not_accept_boolean_hop_cap",lambda:cap_validation(True))
case("cache_hit_does_not_accept_float_hop_cap",lambda:cap_validation(1.0))

def wrong_payload():
    step=k.semantic_step(base,A,[k.Operation("set",A,value=bytearray(b"mutable"))],k.render_exact)
    assert step.output=="red" and not step.transition.accepted and step.transition.store is base
    return step.transition.status
case("malformed_value_proposal_preserves_output_and_state",wrong_payload)

def literal_authority():
    tx=k.prepare(base,[k.Operation("set",A,value="new")])
    r=k.commit(base,tx,authorized=1)
    assert not r.accepted and r.store is base
    return r.status
case("truthy_integer_is_not_policy_authorization",literal_authority)

def prewrite_copy():
    step=k.semantic_step(base,C,[k.Operation("copy",C,source=A)],k.render_exact)
    assert step.output=="UNRESOLVED:missing"
    assert step.transition.accepted and k.resolve(step.transition.store,C).value=="red"
    return {"prewrite_output":step.output,"postwrite_value":k.resolve(step.transition.store,C).value}
case("query_does_not_observe_its_incoming_copy",prewrite_copy)

def empty_value():
    s=put(base,k.Operation("set",A,value=""))
    r=k.resolve(s,A)
    assert r.status=="ok" and k.render_exact(r)==""
    return "valid empty value differs from unresolved"
case("empty_string_value_is_not_missing_or_deleted",empty_value)

def protected_live():
    s=put(base,k.Operation("link",C,source=A))
    r=k.commit(s,k.prepare(s,[k.Operation("set",A,value="green")]),protected=frozenset({C}))
    assert r.accepted and r.store.cell(C)==s.cell(C)
    assert k.resolve(s,C).value=="red" and k.resolve(r.store,C).value=="green"
    return "raw-cell protection intentionally does not protect a live dependent answer"
case("protected_link_cell_does_not_imply_answer_preservation",protected_live)

def equivalent_snapshot():
    _,cache,_=k.cached_resolution(base,A)
    equivalent=k.Store(tuple(base.entries),capacity=base.capacity,epochs=dataclasses.replace(base.epochs))
    r,_,hit=k.cached_resolution(equivalent,A,cache=cache)
    assert r.value=="red"
    return {"hit":hit,"note":"exact equivalent pure-read inputs may safely reuse"}
case("equivalent_reconstructed_pure_snapshot_remains_correct",equivalent_snapshot)

def dependency_coverage():
    r=k.resolve(linked,D,max_links=2)
    assert [x[0] for x in r.dependencies]==[D,C,A]
    missing=put(empty,k.Operation("link",D,source=C))
    r2=k.resolve(missing,D,max_links=2)
    assert [x[0] for x in r2.dependencies]==[D,C] and r2.dependencies[-1][1] is None
    return {"full_chain_dependencies":3,"missing_terminal_recorded":True}
case("resolution_covers_transitive_and_missing_terminal_dependencies",dependency_coverage)

def incompatible_schema():
    s=dataclasses.replace(base,epochs=dataclasses.replace(base.epochs,schema="new-schema"))
    r=k.resolve(s,A)
    assert r.status=="incompatible" and r.value is None
    return r.status
case("schema_epoch_change_does_not_reinterpret_old_payload",incompatible_schema)

def malformed_observation_number(value):
    tx=k.prepare(base,[k.Operation("set",A,value="new")])
    malformed=dataclasses.replace(tx.expected[0][1],revision=value)
    tx=dataclasses.replace(tx,expected=((A,malformed),))
    r=k.commit(base,tx)
    assert not r.accepted and r.store is base
    return r.status
case("boolean_cell_revision_cannot_alias_valid_integer_observation",lambda:malformed_observation_number(True))
case("float_cell_revision_cannot_alias_valid_integer_observation",lambda:malformed_observation_number(1.0))

def malformed_tx_epoch():
    s=dataclasses.replace(base,epochs=dataclasses.replace(base.epochs,authority=1))
    tx=k.prepare(s,[k.Operation("set",A,value="new")])
    tx=dataclasses.replace(tx,epochs=dataclasses.replace(tx.epochs,authority=True))
    r=k.commit(s,tx)
    assert not r.accepted and r.store is s
    return r.status
case("boolean_transaction_authority_epoch_cannot_alias_integer",malformed_tx_epoch)

def bounded_operations():
    calls=[]
    def generated():
        for i in range(1000):
            calls.append(i)
            yield k.Operation("set",("scope",str(i),"value"),value="x")
    try:
        k.prepare(empty,generated())
    except ValueError:
        assert len(calls)<=k.MAX_TRANSACTION_WRITES+1
        return {"elements_pulled":len(calls),"note":"assumes each iterator step terminates"}
    raise AssertionError("write budget failed to reject")
case("operation_count_budget_stops_generator_consumption",bounded_operations)

def bounded_observed():
    calls=[]
    def generated():
        for i in range(1000):
            calls.append(i)
            yield ("scope",str(i),"value")
    try:
        k.prepare(empty,(),observed=generated())
    except ValueError:
        assert len(calls)<=k.MAX_DECLARED_READS+1
        return {"elements_pulled":len(calls)}
    raise AssertionError("dependency budget failed to reject")
case("declared_read_budget_stops_generator_consumption",bounded_observed)

def metadata_bounds():
    oversized="x"*100000
    cases=[lambda:k.Store(epochs=dataclasses.replace(k.Epochs(),model=oversized)),
           lambda:k.Store(((A,k.Cell(1,k.Payload("value",value="x"),oversized)),)),
           lambda:k.Store(capacity=1<<40)]
    assert all(rejected_constructor(fn) for fn in cases)
    return {"oversized_cases_rejected":len(cases)}
case("store_metadata_and_capacity_bounds_are_enforced",metadata_bounds)

def protected_loop():
    step=k.semantic_step(base,A,[k.Operation("set",A,value="new")],k.render_exact,
                         protected=frozenset({A}))
    assert step.output=="red" and not step.transition.accepted
    assert step.transition.store is base
    return step.transition.status
case("one_step_loop_forwards_protected_record_policy",protected_loop)

def prepare_cap_validation():
    for cap in (True,1.0,-1,k.MAX_LINKS+1):
        try:
            k.prepare(base,[k.Operation("set",A,value="new")],max_links=cap)
        except ValueError:
            continue
        raise AssertionError("invalid cap accepted without COPY: "+repr(cap))
    return {"invalid_caps_rejected":4}
case("prepare_validates_hop_cap_even_without_copy",prepare_cap_validation)

parser=argparse.ArgumentParser()
parser.add_argument("--receipt",type=Path,default=ROOT/"review"/"reference_independent_receipt.json")
args=parser.parse_args()
receipt={
 "study":"COG-UNIFIED-001","reviewer":"unified_mechanisms",
 "scope":"independent constructed semantic adversaries; no neural model execution",
 "utc_finished":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 "review_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 "architecture_sha256":hashlib.sha256((ROOT/"ARCHITECTURE.md").read_bytes()).hexdigest(),
 "checks":len(outcomes),"passed":sum(x["status"]=="PASS" for x in outcomes),
 "failed":sum(x["status"]=="FAIL" for x in outcomes),"results":outcomes,
 "trained_models":0,"protected_data_reads":0,"performance_claims":0,
 "limits":["finite constructed cases, not an exhaustive proof","no Python-to-Lean refinement proof",
           "cache object is trusted internal data, not an authenticated external certificate",
           "old revision-only predicate is reconstructed, not an old full-source rerun"]
}
receipt["status"]="PASS" if receipt["failed"]==0 else "FAIL"
args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"status":receipt["status"],"checks":receipt["checks"],"failed":receipt["failed"],
                  "receipt":str(args.receipt),"failures":[x["name"] for x in outcomes if x["status"]=="FAIL"]}))
sys.exit(0 if receipt["failed"]==0 else 1)

