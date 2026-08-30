from casm.benchmark_v1 import (
    TASKS, build_suite, manifest, normalize_answer,
    prompt_hash_from_training_text, score_predictions, suite_digest,
)

EXPECTED_DIGESTS = {
    "dev-core": "c495733cfb3e26098b192a208443206153de9ab32cc0beb9f2e3922946e88d0f",
    "dev-ood": "539e65af21bba0b6f542037073e3199c89cd6672169c90ea767d2d9be26c4ef5",
    "holdout-core": "112a45d5c5ddd9a28e1746995980129e2cfce6e549c8a2b0bc19a892554f4eca",
    "holdout-ood": "a7f9618d1d7c8d03143a7b3368cfee3f3e50559599b1223b20001b8cbe12c91f",
}


def test_suite_digests_are_frozen():
    for name, expected in EXPECTED_DIGESTS.items():
        assert suite_digest(build_suite(name)) == expected


def test_cross_suite_prompt_overlap_is_zero():
    m = manifest()
    assert set(m["tasks"]) == set(TASKS)
    assert all(v == 0 for v in m["cross_suite_prompt_overlap"].values())


def test_gold_is_not_in_visible_prompt_and_state_is_complete():
    for name in EXPECTED_DIGESTS:
        for x in build_suite(name):
            assert x.prompt.endswith("answer ")
            assert not x.prompt.endswith("answer " + x.answer)
            if x.task == "state":
                assert "\ninitial " in x.prompt
                assert x.metadata["verified_final"] == x.answer


def test_graph_is_exactly_balanced_and_verified():
    for name in EXPECTED_DIGESTS:
        graph = [x for x in build_suite(name) if x.task == "graph"]
        assert sum(x.answer == "yes" for x in graph) == len(graph) // 2
        assert sum(x.answer == "no" for x in graph) == len(graph) // 2
        for x in graph:
            assert bool(x.metadata["verified_reachable"]) == (x.answer == "yes")


def test_ood_domains_are_structurally_disjoint():
    core = build_suite("dev-core")
    ood = build_suite("dev-ood")
    def vals(rows, task, key):
        return {x.metadata[key] for x in rows if x.task == task}
    assert vals(core, "assoc", "n_keys") == {12}
    assert vals(ood, "assoc", "n_keys") == {24}
    assert vals(core, "state", "n_events") == {12}
    assert vals(ood, "state", "n_events") == {24}
    assert vals(core, "graph", "n_nodes") == {10}
    assert vals(ood, "graph", "n_nodes") == {14}
    assert vals(core, "reverse", "length") == {14}
    assert vals(ood, "reverse", "length") == {28}
    assert vals(core, "rule", "mul") <= {2, 3}
    assert vals(ood, "rule", "mul") <= {4, 5}
    assert vals(core, "rule", "mul").isdisjoint(vals(ood, "rule", "mul"))


def test_numeric_normalization_is_strict_semantic_not_commentary():
    assert normalize_answer("arith", "0042") == "42"
    assert normalize_answer("arith", "42 because") == "42 because"
    assert normalize_answer("graph", " YES \n") == "yes"


def test_majority_graph_collapse_scores_zero_adjusted():
    rows = build_suite("dev-core")
    predictions = {x.case_id: ("yes" if x.task == "graph" else x.answer) for x in rows}
    scored = score_predictions(rows, predictions)
    assert abs(scored["per_task"]["graph"]["raw_exact"] - 0.5) < 1e-12
    assert abs(scored["per_task"]["graph"]["adjusted_exact"]) < 1e-12


def test_training_hash_strips_gold_answer():
    a = prompt_hash_from_training_text("task arithmetic\ncompute 1+1\nanswer 2")
    b = prompt_hash_from_training_text("task arithmetic\ncompute 1+1\nanswer 999")
    assert a == b
