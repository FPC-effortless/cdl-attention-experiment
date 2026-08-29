import random
import re

from casm.data import graph_reachability


def parse_graph(ex):
    lines = ex.text.splitlines()
    edge_line = next(x for x in lines if x.startswith("edges "))[6:]
    query_line = next(x for x in lines if x.startswith("reachable ")).split()
    src, dst = query_line[1], query_line[2]
    edges = []
    for token in edge_line.split():
        a, b = token.split("->")
        edges.append((a, b))
    return src, dst, edges


def reachable(src, dst, edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    seen = {src}
    stack = [src]
    while stack:
        x = stack.pop()
        if x == dst:
            return True
        for y in adj.get(x, []):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def test_graph_generator_has_both_labels_and_correct_reachability():
    rng = random.Random(20260829)
    labels = set()
    for hard in (False, True):
        for _ in range(200):
            ex = graph_reachability(rng, hard=hard)
            labels.add(ex.answer)
            src, dst, edges = parse_graph(ex)
            assert reachable(src, dst, edges) == (ex.answer == "yes")
    assert labels == {"yes", "no"}
