"""LRR-007 controlled attention-free address retrieval study.

Frozen per-seed results, actual TEST token positions, and hard-write diagnostics.
No threshold is calibrated for abstention; unknown and absent are diagnostics.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .lrr005_register import CausalRoleRegister, ID, FAMILIES
from .lrr007_dual_credit import (
    fit_arm, training_triple, heldout_example
)
from .run_lrr005 import SymbolicRoleIndex, known_absent, unknown_relation


@torch.no_grad()
def hard_diagnostics(model: CausalRoleRegister, case) -> dict:
    """Actual hard-index state trace; evaluator labels used only for diagnosis."""
    words=case.words
    x=torch.tensor([[ID.get(word,1) for word in words]],dtype=torch.long)
    h=model.token_features(x)
    gates=model.role(h).sigmoid()[0].tolist()
    codes=model.code(h)[0].argmax(-1).tolist()
    entity=relation=None
    index={}
    gold_key=None
    wrote_gold=False
    writes=nonvalue=question_writes=0
    record_keys={}
    clause_values={f[-1]:f for f in case.facts}
    value_position_set=set(clause_values)
    query_start=words.index("what")
    for j,word in enumerate(words[:-1]):
        key=(entity,relation) if entity is not None and relation is not None else None
        if j==case.answer:
            gold_key=key
            wrote_gold=key is not None and gates[j][2]>.5
        if j in value_position_set:
            record_keys[j]=key if (gates[j][2]>.5) else None
        if gates[j][2]>.5 and key is not None:
            index[key]=j
            writes+=1
            nonvalue+=int(j not in value_position_set)
            question_writes+=int(j>=query_start)
        if gates[j][0]>.5:
            entity=word
        if gates[j][1]>.5:
            relation=codes[j]
    query_key=(entity,relation)
    selected=index.get(query_key)
    from_router,rt=model.route(words)
    if selected!=from_router or writes!=rt["writes"]:
        raise AssertionError("trace disagrees with audited source hard router")
    if selected==case.answer:
        failure="correct"
    elif selected is None:
        failure="no_candidate"
    elif gold_key==query_key and wrote_gold:
        failure="collision_or_overwrite"
    elif not wrote_gold:
        failure="gold_not_written"
    else:
        failure="address_mismatch"
    collisions=0
    items=[(f,record_keys.get(f[-1])) for f in case.facts
           if record_keys.get(f[-1]) is not None]
    for i,(fact,key) in enumerate(items):
        for other,other_key in items[i+1:]:
            if fact[0]==other[0] and fact[1]!=other[1] and key==other_key:
                collisions+=1
    return {
        "selected_index":selected,"failure":failure,"index_probes":1,
        "gold_admitted":wrote_gold, "query_gold_address_equal":gold_key==query_key,
        "writes":writes,"nonvalue_writes":nonvalue,
        "question_writes":question_writes,
        "distinct_addresses":len(index),
        "true_different_relation_same_address_pairs":collisions,
    }


def train_symbolic(seed, *, steps, triples_per_update):
    store=SymbolicRoleIndex()
    for step in range(steps):
        for i in range(triples_per_update):
            for ex in training_triple(seed*1_000_000+step*1_000+i,step,i):
                store.add_example(ex)
    return store


def evaluate(model, symbolic, *, seed, facts, layout, trials):
    cases=[
        heldout_example(
            70_000_000+seed*1_000_000+facts*1000+i,
            facts,split="test",layout=layout)
        for i in range(trials)
    ]
    correct=0
    reference=0
    failures=Counter()
    total_writes=total_nonvalue=question_writes=0
    distinct=0
    gold_admitted=address_matches=0
    semantic_collisions=0
    for case in cases:
        trace=hard_diagnostics(model,case)
        correct+=int(trace["selected_index"]==case.answer)
        reference+=int(symbolic.route(case.words)==case.answer)
        failures[trace["failure"]]+=1
        total_writes+=trace["writes"]
        total_nonvalue+=trace["nonvalue_writes"]
        question_writes+=trace["question_writes"]
        distinct+=trace["distinct_addresses"]
        gold_admitted+=trace["gold_admitted"]
        address_matches+=trace["query_gold_address_equal"]
        semantic_collisions+=trace["true_different_relation_same_address_pairs"]
    result={
        "trials":trials,
        "correct":correct,
        "recall_at_1":correct/trials,
        "symbolic_correct":reference,
        "symbolic_recall":reference/trials,
        "failures":dict(failures),
        "mean_writes":total_writes/trials,
        "mean_nonvalue_writes":total_nonvalue/trials,
        "mean_question_writes":question_writes/trials,
        "mean_distinct_addresses":distinct/trials,
        "gold_admission_rate":gold_admitted/trials,
        "query_gold_address_match_rate":address_matches/trials,
        "semantic_hard_address_collision_pairs_per_doc":semantic_collisions/trials,
    }
    if layout=="regular":
        absent=[known_absent(ex) for ex in cases]
        unknown=[unknown_relation(ex) for ex in cases]
        result["known_absent_false_answers"]=sum(
            model.route(words)[0] is not None for words in absent)
        result["unknown_predicate_false_answers"]=sum(
            model.route(words)[0] is not None for words in unknown)
    return result


def run(*, seeds=(51,52,53,54,55), facts=(7,32,128),
        trials=64, steps=500, triples_per_update=16,
        checkpoint_dir=None):
    torch.set_num_threads(1)
    arms=("negative_only","two_sided","energy_rank")
    report={
        "experiment":"LRR-007",
        "status":"controlled supervised address consistency, no LM attention replacement",
        "device":"CPU single-thread",
        "python_version":platform.python_version(),
        "torch_version":torch.__version__,
        "trainable_router_parameters":4748,
        "method":"same-answer positive alias and different-answer negative relation credit",
        "seeds":list(seeds),"facts":list(facts),
        "trials_per_seed_fact_layout":trials,
        "updates_per_arm":steps,"triples_per_update":triples_per_update,
        "training_questions_per_seed_arm":steps*triples_per_update*3,
        "training_forbidden_pairs":[[0,3],[2,0],[1,2],[3,0]],
        "dev_pair":[2,0],"confirmatory_test_pair":[0,3],
        "arms":list(arms),"train":[],"results":[],
    }
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        started=time.perf_counter()
        symbolic=train_symbolic(seed,steps=steps,
                                triples_per_update=triples_per_update)
        symbolic_time=time.perf_counter()-started
        if symbolic.uf.count()!=4:
            raise AssertionError("symbolic control failed to infer 4 groups")
        for arm in arms:
            model,metadata=fit_arm(seed,arm,steps=steps,
                                    triples_per_update=triples_per_update)
            if metadata["trainable_parameters"]!=4748:
                raise AssertionError("unexpected parameter budget")
            report["train"].append({
                **metadata,"symbolic_equivalence_groups":symbolic.uf.count(),
                "symbolic_training_seconds":symbolic_time,
            })
            if checkpoint_dir is not None:
                torch.save(model.state_dict(),
                           Path(checkpoint_dir)/f"lrr007_{arm}_seed{seed}.pt")
            for n in facts:
                for layout in ("regular","shift","novel_fillers"):
                    result=evaluate(model,symbolic,seed=seed,n=n,
                                    facts=n,layout=layout,trials=trials) if False else evaluate(
                                        model,symbolic,seed=seed,
                                        facts=n,layout=layout,trials=trials)
                    report["results"].append({
                        "arm":arm,"seed":seed,"facts":n,
                        "layout":layout,"outcomes":result
                    })
    def cell(arm,seed,n=32,layout="regular"):
        matches=[x["outcomes"] for x in report["results"]
                 if x["arm"]==arm and x["seed"]==seed and
                    x["facts"]==n and x["layout"]==layout]
        if len(matches)!=1:
            raise AssertionError("missing or duplicated evaluation result")
        return matches[0]
    base=[cell("negative_only",seed)["recall_at_1"] for seed in seeds]
    main=[cell("two_sided",seed)["recall_at_1"] for seed in seeds]
    margin=[cell("energy_rank",seed)["recall_at_1"] for seed in seeds]
    delta=[b-a for a,b in zip(base,main)]
    shift=[cell("two_sided",seed,layout="shift")["recall_at_1"] for seed in seeds]
    novel=[cell("two_sided",seed,layout="novel_fillers")["recall_at_1"] for seed in seeds]
    symbolic=[cell("two_sided",seed)["symbolic_recall"] for seed in seeds]
    report["decision"]={
        "negative_only_by_seed":base,
        "two_sided_by_seed":main,
        "energy_rank_secondary_by_seed":margin,
        "paired_deltas":delta,
        "negative_only_mean":statistics.mean(base),
        "two_sided_mean":statistics.mean(main),
        "energy_rank_secondary_mean":statistics.mean(margin),
        "known_filler_shift_mean":statistics.mean(shift),
        "novel_filler_mean":statistics.mean(novel),
        "symbolic_ceiling_mean":statistics.mean(symbolic),
        "G1_capability":statistics.mean(main)>=.85 and sum(x>=.80 for x in main)>=4,
        "G2_attributable_uplift":statistics.mean(delta)>=.10 and min(delta)>=-.05,
        "G3_syntax_shift":statistics.mean(shift)>=.75,
        "G4_attention_replacement":"UNTESTED, requires capable matched LM baseline",
        "no_abstention_gate_on_untrained_router":True,
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="51,52,53,54,55")
    p.add_argument("--lengths",default="7,32,128")
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--triples",type=int,default=16)
    p.add_argument("--trials",type=int,default=64)
    p.add_argument("--output",type=Path,default=Path("/tmp/lrr007-results.json"))
    p.add_argument("--checkpoints",type=Path,default=Path("/tmp/lrr007-checkpoints"))
    a=p.parse_args()
    result=run(seeds=tuple(int(x) for x in a.seeds.split(",")),
               facts=tuple(int(x) for x in a.lengths.split(",")),
               steps=a.steps,triples_per_update=a.triples,trials=a.trials,
               checkpoint_dir=a.checkpoints)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"decision":result["decision"],
                      "fit_count":len(result["train"]),
                      "evaluation_cell_count":len(result["results"]),
                      "output":str(a.output)},indent=2))


if __name__=="__main__":
    main()
