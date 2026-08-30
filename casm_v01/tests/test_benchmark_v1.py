from casm.benchmark_v1 import (
    TASKS, build_suite, manifest, normalize_answer,
    prompt_hash_from_training_text, score_predictions, suite_digest,
)

EXPECTED_DIGESTS = {
    "dev-core": "a08e78746d3f3f93e05137f701fd7f6c734c5437beccf5ed0d0a2c1dacda5a0a",
    "dev-ood": "67880ecad3fe300f724a7e27e7b3c873aae44dd09e526b12a998cba0a8fbf52e",
    "holdout-core": "7af0e22f77d3df17761b124fbb6eb8a19560d94bbe3f898fd321a68ce862666f",
    "holdout-ood": "627ed850f55cef9af16318120ec54ddb14ec7b9ad620d178a4904265aa13bcef",
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
                expected_min = 6 if name.endswith("ood") else 3
                assert x.metadata["query_updates"] >= expected_min


def test_graph_is_exactly_balanced_and_verified():
    for name in EXPECTED_DIGESTS:
        graph = [x for x in build_suite(name) if x.task == "graph"]
        assert sum(x.answer == "yes" for x in graph) == len(graph) // 2
        assert sum(x.answer == "no" for x in graph) == len(graph) // 2
        for x in graph:
            assert bool(x.metadata["verified_reachable"]) == (x.answer == "yes")


def test_graph_counterfactual_pairs_match_except_for_one_edge_swap():
    for name in EXPECTED_DIGESTS:
        graph = [x for x in build_suite(name) if x.task == "graph"]
        pairs = {}
        for x in graph:
            pairs.setdefault(x.metadata["pair_id"], {})[x.metadata["variant"]] = x
        assert len(pairs) * 2 == len(graph)
        for pair in pairs.values():
            assert set(pair) == {"positive", "negative"}
            pos, neg = pair["positive"], pair["negative"]
            assert pos.answer == "yes" and neg.answer == "no"
            assert pos.metadata["src"] == neg.metadata["src"]
            assert pos.metadata["dst"] == neg.metadata["dst"]
            pe = set(map(tuple, pos.metadata["edges"]))
            ne = set(map(tuple, neg.metadata["edges"]))
            assert len(pe) == len(ne)
            assert len(pe - ne) == 1 and len(ne - pe) == 1
            minimum = 6 if name.endswith("ood") else 4
            assert pos.metadata["shortest_path"] >= minimum


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
