"""Constructed semantic falsifiers only; no trained model or benchmark data."""
import itertools
import json
from dataclasses import replace
from pathlib import Path
from semantic_kernel import (
    Cell, Epochs, MAX_REVISION, Operation, Payload, Store, Transaction,
    affine_apply, affine_compose, cached_resolution, commit, prepare,
    render_exact, resolve, semantic_step,
)

A = ("scope", "alice", "holder")
B = ("scope", "bob", "holder")
C = ("scope", "carol", "holder")
D = ("scope", "dave", "holder")
results = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    results.append({"name": name, "status": "PASS"})


def apply(store, *ops, **kwargs):
    result = commit(store, prepare(store, ops), **kwargs)
    if not result.accepted:
        raise AssertionError(result.status)
    return result.store


empty = Store(capacity=8)
s = apply(empty, Operation("set", A, value="red"), Operation("set", B, value="blue"))
check("set_reads_exactly", resolve(s, A).value == "red")
overwritten = apply(s, Operation("set", A, value="green"))
check("overwrite_replaces_old_value", resolve(overwritten, A).value == "green")
check("unwritten_cell_identical", overwritten.cell(B) == s.cell(B))
check("old_snapshot_immutable", resolve(s, A).value == "red" and not empty.entries)
check("revision_strictly_increases", overwritten.revision(A) == s.revision(A) + 1)

linked = apply(s, Operation("link", C, source=A), Operation("copy", D, source=A))
changed = apply(linked, Operation("set", A, value="green"))
check("copy_is_snapshot", resolve(changed, D).value == "red")
check("link_is_live", resolve(changed, C).value == "green")
check("raw_cell_preservation_does_not_preserve_link_answer",
      changed.cell(C) == linked.cell(C) and resolve(changed, C).value != resolve(linked, C).value)
deleted = apply(changed, Operation("delete", A))
check("delete_is_explicit", resolve(deleted, A).status == "deleted")
check("dangling_link_is_not_a_value", resolve(deleted, C).status == "deleted")
check("deletion_does_not_reset_version", deleted.revision(A) == changed.revision(A) + 1)

read, cache, hit = cached_resolution(linked, C)
check("initial_cache_miss", not hit and read.value == "red")
read, _, hit = cached_resolution(changed, C, cache=cache)
check("transitive_source_update_invalidates_cache", not hit and read.value == "green")
unrelated = apply(linked, Operation("set", B, value="purple"))
read, _, hit = cached_resolution(unrelated, C, cache=cache)
check("unrelated_change_keeps_pure_read_cache", hit and read.value == "red")
_, missing_cache, _ = cached_resolution(empty, A)
read, _, hit = cached_resolution(s, A, cache=missing_cache)
check("absent_dependency_insertion_invalidates", not hit and read.value == "red")
for field in ["stream", "authority", "policy", "schema", "model", "tokenizer"]:
    old = getattr(linked.epochs, field)
    new = old + 1 if isinstance(old, int) else old + "-new"
    epoch_store = replace(linked, epochs=replace(linked.epochs, **{field: new}))
    _, _, hit = cached_resolution(epoch_store, C, cache=cache)
    check("cache_epoch_" + field, not hit)

copy_tx = prepare(linked, [Operation("copy", B, source=C)])
rejected = commit(changed, copy_tx)
check("stale_transitive_copy_is_rejected", not rejected.accepted and rejected.store is changed)
rejected = commit(s, prepare(s, [Operation("set", A, value="bad")]), authorized=False)
check("rejected_authority_preserves_whole_snapshot", not rejected.accepted and rejected.store is s)
rejected = commit(s, prepare(s, [Operation("set", A, value="bad")]), protected=frozenset({A}))
check("protected_key_rejected", not rejected.accepted and rejected.store is s)
tx = prepare(s, [Operation("set", A, value="one"), Operation("set", B, value="two")])
rejected = commit(overwritten, tx)
check("failed_batch_has_no_partial_write", not rejected.accepted and rejected.store is overwritten)
for field in ["authority", "policy", "schema", "model", "tokenizer", "stream"]:
    old = getattr(s.epochs, field)
    new = old + 1 if isinstance(old, int) else old + "-new"
    new_store = replace(s, epochs=replace(s.epochs, **{field: new}))
    check("transaction_epoch_" + field, not commit(new_store, tx).accepted)

tx_a = prepare(s, [Operation("set", A, value="one")])
tx_b = prepare(s, [Operation("set", B, value="two")])
ab = commit(commit(s, tx_a).store, tx_b)
ba = commit(commit(s, tx_b).store, tx_a)
check("independent_transactions_commute", ab.accepted and ba.accepted and ab.store == ba.store)
read_a_write_b = prepare(s, [Operation("set", B, value="off")], observed=[A])
read_b_write_a = prepare(s, [Operation("set", A, value="off")], observed=[B])
one = commit(s, read_a_write_b)
two = commit(one.store, read_b_write_a)
check("read_write_conflict_rejects_write_skew", one.accepted and not two.accepted)

cycle = apply(empty, Operation("link", A, source=B), Operation("link", B, source=A))
check("cycle_explicit", resolve(cycle, A).status == "cycle")
chain = apply(s, Operation("link", C, source=A), Operation("link", D, source=C))
check("hop_budget_rejects_long_chain", resolve(chain, D, 1).status == "hop_limit")
check("sufficient_hop_budget_resolves", resolve(chain, D, 2).value == "red")
one_slot = Store(capacity=1)
full = apply(one_slot, Operation("set", A, value="red"))
rejected = commit(full, prepare(full, [Operation("set", B, value="blue")]))
check("capacity_exhaustion_preserves_old_facts", rejected.status == "capacity_exhausted" and rejected.store is full)
max_store = Store(((A, Cell(MAX_REVISION, Payload("value", value="red"), "typed-string-v1")),))
rejected = commit(max_store, prepare(max_store, [Operation("set", A, value="green")]))
check("version_overflow_rejects_without_wrap", rejected.status == "revision_exhausted" and rejected.store is max_store)
malformed = Transaction(s.epochs, (), ((A, Payload("value", value="bad")),))
check("write_without_version_guard_rejected", not commit(s, malformed).accepted)

step = semantic_step(s, A, [Operation("set", A, value="green")], render_exact)
check("read_before_current_write", step.output == "red" and resolve(step.transition.store, A).value == "green")
bad_grounding = semantic_step(s, B, (), render_exact)
check("perfect_store_does_not_fix_wrong_predicted_key", bad_grounding.output == "blue" and bad_grounding.output != resolve(s, A).value)
wrong_decoder = semantic_step(s, A, (), lambda _: "wrong")
check("correct_retrieval_does_not_force_neural_output", wrong_decoder.output == "wrong")

transforms = list(itertools.product(range(3), repeat=2))
for f, g, h in itertools.product(transforms, repeat=3):
    if affine_compose(affine_compose(f, g), h) != affine_compose(f, affine_compose(g, h)):
        raise AssertionError("affine associativity")
check("729_exact_affine_associativity_instances", True)
for f, g, initial in itertools.product(transforms, transforms, range(4)):
    if affine_apply(affine_compose(f, g), initial) != affine_apply(f, affine_apply(g, initial)):
        raise AssertionError("affine action")
check("324_exact_affine_action_instances", True)

# No performance/semantic claims arise from these deliberately finite witnesses.
old_model = lambda x: 0
edited_model = lambda x: 1 if x == 10 else 0
check("finite_regression_can_miss_new_error",
      all(old_model(x) == edited_model(x) for x in range(10))
      and old_model(10) != edited_model(10))
candidate_keys = [A, B]
check("bounded_candidates_can_omit_required_record", C not in candidate_keys and resolve(linked, C).status == "ok")
check("fixed_state_collision_witness", (0 % 2 == 2 % 2) and 0 != 2)
# Same exact record/query but a decoder that also consumes h can change output.
neural_read = lambda h, value: value if h == 0 else "other"
check("neural_output_requires_causal_state_identity",
      neural_read(0, "red") != neural_read(1, "red"))

report = {"study": "COG-UNIFIED-001", "kind": "constructed semantic checks",
          "checks": len(results), "status": "PASS", "trained_models": 0,
          "protected_data_reads": 0, "latency_claims": 0,
          "limits": ["not Python-to-Lean refinement", "not learned binding",
                     "not distributed or crash-consistent storage", "not real workload accuracy"],
          "results": results}
target = Path(__file__).resolve().parents[1] / "checks" / "reference_checks.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "checks": len(results), "receipt": str(target)}))
